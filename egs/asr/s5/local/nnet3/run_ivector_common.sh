#!/bin/bash

set -euo pipefail

# This script is modified based on mini_librispeech/s5/local/nnet3/run_ivector_common.sh

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and
# iVector-related parts of the script.  See those scripts for examples
# of usage.

stage=0
online=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

outdir=$1
gmm_dir=$outdir/exp/tri5
ali_dir=${gmm_dir}_sp_ali
train_set=train
train_set_sp=${train_set}_sp
path_train_set_sp=${outdir}/data/${train_set_sp}
test_set=test
nnet3_affix=

for f in $outdir/data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

online_affix=
if [ $online = true ]; then
  online_affix=_online
fi

if [ $stage -le 1 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    $outdir/data/${train_set} \
    $path_train_set_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 70 \
    $path_train_set_sp \
    $outdir/exp/make_mfcc/$train_set_sp \
    $outdir/mfcc_perturbed || exit 1;
  steps/compute_cmvn_stats.sh \
    $path_train_set_sp \
    $outdir/exp/make_mfcc/$train_set_sp \
    $outdir/mfcc_perturbed || exit 1;
  utils/fix_data_dir.sh \
    $path_train_set_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    $path_train_set_sp \
    $outdir/data/lang \
    $gmm_dir \
    $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=$outdir/mfcc_perturbed_hires$online_affix

  for datadir in $train_set_sp $test_set; do
    utils/copy_data_dir.sh \
      $outdir/data/$datadir \
      $outdir/data/${datadir}_hires$online_affix
  done
  
  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh \
    $outdir/data/${train_set}_sp_hires$online_affix || exit 1;

  for datadir in ${train_set}_sp $test_set; do
    steps/make_mfcc_pitch$online_affix.sh --nj 20 \
      --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" \
      $outdir/data/${datadir}_hires$online_affix \
      $outdir/exp/make_hires/$datadir \
      $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh \
      $outdir/data/${datadir}_hires$online_affix \
      $outdir/exp/make_hires/$datadir \
      $mfccdir || exit 1;
    utils/fix_data_dir.sh \
      $outdir/data/${datadir}_hires$online_affix || exit 1;
    # create MFCC data dir without pitch to extract iVector
    utils/data/limit_feature_dim.sh 0:39 \
      $outdir/data/${datadir}_hires$online_affix \
      $outdir/data/${datadir}_hires_nopitch || exit 1;
    steps/compute_cmvn_stats.sh \
      $outdir/data/${datadir}_hires_nopitch \
      $outdir/exp/make_hires/$datadir \
      $mfccdir || exit 1;
  done
fi

diag_ubm=$outdir/exp/nnet3${nnet3_affix}/diag_ubm
if [ $stage -le 4 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."
  # We'll use about a quarter of the data.
  mkdir -p $diag_ubm

  pca_transform=$outdir/exp/nnet3${nnet3_affix}/pca_transform
  nopitch_subset=${diag_ubm}/${train_set_sp}_hires_nopitch_subset
  hires_nopitch=$outdir/data/${train_set_sp}_hires_nopitch

  num_utts_total=$(wc -l < ${hires_nopitch}/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh \
      $hires_nopitch \
      $num_utts \
      $nopitch_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
      $nopitch_subset \
      $pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads 8 \
    $nopitch_subset \
    512 \
    $pca_transform \
    $diag_ubm
fi

extractor=$outdir/exp/nnet3${nnet3_affix}/extractor
if [ $stage -le 5 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
     $hires_nopitch \
     $diag_ubm \
     $extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: extract ivectors"
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  ivectordir=$outdir/exp/nnet3${nnet3_affix}/ivectors_${train_set_sp}

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  hires_nopitch_max2=${ivectordir}/${train_set_sp}_hires_nopitch_max2
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $hires_nopitch \
    $hires_nopitch_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    $hires_nopitch_max2 \
    $extractor \
    $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      $outdir/data/${test_set}_hires_nopitch \
      $extractor \
      $outdir/exp/nnet3${nnet3_affix}/ivectors_${test_set}
fi
echo "$0 successed"
exit 0
