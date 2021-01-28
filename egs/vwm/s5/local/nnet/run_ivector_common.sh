#!/bin/bash

set -e -o pipefail


# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.


stage=0
min_seg_len=1.55 # min length in seconds... we do this because chain training
                 # will discard segments shorter than 1.5 seconds. Must remain in sync
                 # with the same option given to prepare_lores_feats_and_alignments.sh
train_set=""     # you might set this to e.g. train_960
gmm_dir=""           # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.
num_threads_ubm=32
num_processes=4
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.
out_dir=""
nj=64
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

ali_dir=${gmm_dir}_ali_${train_set}_sp_comb

for f in $out_dir/data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
  # _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
	  $out_dir/data/${train_set} \
	  $out_dir/data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
#  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj $out_dir/data/${train_set}_sp || exit 1;
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
	  $out_dir/data/${train_set}_sp

  steps/compute_cmvn_stats.sh $out_dir/data/${train_set}_sp || exit 1;
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh $out_dir/data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: combining short segments of low-resolution speed-perturbed MFCC data"
  src=$out_dir/data/${train_set}_sp
  dest=$out_dir/data/${train_set}_sp_comb
  utils/data/combine_short_segments.sh $src $min_seg_len $dest
  # re-use the CMVN stats from the source directory, since it seems to be slow to
  # re-compute them after concatenating short segments.
  cp $src/cmvn.scp $dest/
  utils/fix_data_dir.sh $dest

  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed, short-segment-combined low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $dest $out_dir/data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  echo "$0: creating high-resolution MFCC features"
#  for datadir in ${train_set}_sp test_clean test_other dev_clean dev_other; do
  for datadir in ${train_set}_sp; do
    utils/copy_data_dir.sh \
	    $out_dir/data/$datadir \
	    $out_dir/data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh $out_dir/data/${train_set}_sp_hires

#  for datadir in ${train_set}_sp test_clean test_other dev_clean dev_other; do
  for datadir in ${train_set}_sp; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" \
      $out_dir/data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh $out_dir/data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh $out_dir/data/${datadir}_hires
  done

  # now create some data subsets.
  # mixed is the clean+other data.
  # 30k is 1/10 of the data (around 100 hours), 60k is 1/5th of it (around 200 hours).
  utils/subset_data_dir.sh \
	  $out_dir/data/${train_set}_sp_hires 30000 \
	  $out_dir/data/${train_set}_sp_mixed_hires_30k
  utils/subset_data_dir.sh \
	  $out_dir/data/${train_set}_sp_hires 60000 \
	  $out_dir/data/${train_set}_sp_mixed_hires_60k
fi

if [ $stage -le 4 ]; then
  echo "$0: combining short segments of speed-perturbed high-resolution MFCC training data"
  # we have to combine short segments or we won't be able to train chain models
  # on those segments.
  src=$out_dir/data/${train_set}_sp_hires
  dest=$out_dir/data/${train_set}_sp_hires_comb
  utils/data/combine_short_segments.sh $src $min_seg_len $dest
  # just copy over the CMVN to avoid having to recompute it.
  cp $src/cmvn.scp $dest/
  utils/fix_data_dir.sh $dest
fi

if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We align a subset of training data for
  # this purpose.
  echo "$0: aligning a subset of training data."
  utils/subset_data_dir.sh --utt-list <(awk '{print $1}' \
    $out_dir/data/${train_set}_sp_mixed_hires_30k/utt2spk) \
    $out_dir/data/${train_set}_sp \
    $out_dir/data/${train_set}_sp_30k

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $out_dir/data/${train_set}_sp_30k \
    $out_dir/data/lang \
    $gmm_dir \
    $out_dir/exp/tri5a_cleaned_ali_30k
fi

if [ $stage -le 6 ]; then
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
    echo "$0: training a system on the hires subset data for its LDA+MLLT transform, in order to produce the diagonal GMM."
  if [ -e $out_dir/exp/nnet3${nnet3_affix}/tri6a/final.mdl ]; then
    # we don't want to overwrite old stuff, ask the user to delete it.
    echo "$0: exp/nnet3${nnet3_affix}/tri6a/final.mdl already exists: "
    echo " ... please delete and then rerun, or use a later --stage option."
    exit 1;
  fi
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 $out_dir/data/${train_set}_sp_mixed_hires_30k \
    $out_dir/data/lang \
    $out_dir/exp/tri5a_cleaned_ali_30k \
    $out_dir/exp/nnet3${nnet3_affix}/tri6a
fi


if [ $stage -le 7 ]; then
  echo "$0: using the subset of data to train the diagonal UBM."

  mkdir -p $out_dir/exp/nnet3${nnet3_affix}/diag_ubm
  # To train a diagonal UBM we don't need very much data, so use a small subset
  # (actually, it's not that small: still around 100 hours).
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    $out_dir/data/${train_set}_sp_mixed_hires_30k 512 \
    $out_dir/exp/nnet3${nnet3_affix}/tri6a \
    $out_dir/exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 8 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 60k subset (about one fifth of the data, or 200 hours).
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj \
    --num-processes $num_processes \
    --ivector-dim 30 \
    $out_dir/data/${train_set}_sp_mixed_hires_60k \
    $out_dir/exp/nnet3${nnet3_affix}/diag_ubm \
    $out_dir/exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 9 ]; then
  echo "$0: extracting iVectors for training data"
  ivectordir=$out_dir/exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker. this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $out_dir/data/${train_set}_sp_hires_comb \
    ${ivectordir}/${train_set}_sp_hires_comb_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${ivectordir}/${train_set}_sp_hires_comb_max2 \
    $out_dir/exp/nnet3${nnet3_affix}/extractor \
    $ivectordir || exit 1;
fi

#if [ $stage -le 10 ]; then
#  echo "$0: extracting iVectors for dev and test data"
#  for data in test_clean test_other dev_clean dev_other; do
#    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
#      $out_dir/data/${data}_hires \
#      $out_dir/exp/nnet3${nnet3_affix}/extractor \
#      $out_dir/exp/nnet3${nnet3_affix}/ivectors_${data}_hires || exit 1;
#  done
#fi

exit 0;
