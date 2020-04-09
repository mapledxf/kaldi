#!/bin/bash
# Copyright    2017   Johns Hopkins University (Author: Daniel Povey)
#              2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#              2018   Ewald Enzinger
#              2018   David Snyder
# Apache 2.0.
#
# This is an x-vector-based recipe for Speakers in the Wild (SITW).
# It is based on "X-vectors: Robust DNN Embeddings for Speaker Recognition"
# by Snyder et al.  The recipe uses augmented VoxCeleb 1 and 2 for training.
# The augmentation consists of MUSAN noises, music, and babble and
# reverberation from the Room Impulse Response and Noise Database.  Note that
# there are 60 speakers in VoxCeleb 1 that overlap with our evaluation
# dataset, SITW.  The recipe removes those 60 speakers prior to training.
# See ../README.txt for more info on data required.  The results are reported
# in terms of EER and minDCF, and are inline in the comments below.

. ./cmd.sh
. ./path.sh
set -e

openslr_aidatatang=/home/data/xfding/dataset/asr/aidatatang_200zh
openslr_aishell=/home/data/xfding/dataset/asr/aishell/data_aishell
openslr_magicdata=/home/data/xfding/dataset/asr/magicdata
openslr_primewords=/home/data/xfding/dataset/asr/primewords_md_2018_set1
openslr_stcmds=/home/data/xfding/dataset/asr/ST-CMDS-20170001_1-OS
openslr_thchs=/home/data/xfding/dataset/asr/thchs30/data_thchs30

vwm_noisy_48h_src=/home/data/xfding/dataset/asr/noisy-48h
vwm_quite_30h_src=/home/data/xfding/dataset/asr/quite-30h
vwm_noisy_48h_out=/home/data/xfding/train_dataset/asr/noisy-48h
vwm_quite_30h_out=/home/data/xfding/train_dataset/asr/quite-30h

out_dir=/home/data/xfding/train_result/xvector

rirs_dir=/home/data/xfding/dataset/RIRS_NOISES

mfccdir=$out_dir/mfcc
vaddir=$out_dir/vad

nnet_dir=$out_dir/exp/xvector_nnet_1a

dev_trials_core=$out_dir/data/dev_test/trials/core-core.lst
eval_trials_core=$out_dir/data/eval_test/trials/core-core.lst

stage=0

decode=false

. utils/parse_options.sh

if [ $stage -le 0 ]; then
	echo "stage 0: Data Prepare"
        local/vwm_data_prep.sh $vwm_noisy_48h_src $vwm_noisy_48h_out $out_dir/data/vwm_noisy_48h || exit 1;
        local/vwm_data_prep.sh $vwm_quite_30h_src $vwm_quite_30h_out $out_dir/data/vwm_quite-30h || exit 1;

        local/aidatatang_data_prep.sh $openslr_aidatatang $out_dir/data/aidatatang || exit 1;
        local/aishell_data_prep.sh $openslr_aishell $out_dir/data/aishell || exit 1;
        local/thchs-30_data_prep.sh $openslr_thchs $out_dir/data/thchs || exit 1;
        local/magicdata_data_prep.sh $openslr_magicdata $out_dir/data/magicdata || exit 1;
        local/primewords_data_prep.sh $openslr_primewords $out_dir/data/primewords || exit 1;
        local/stcmds_data_prep.sh $openslr_stcmds $out_dir/data/stcmds || exit 1;
        
	utils/combine_data.sh $out_dir/data/train_combined \
                $out_dir/data/{vwm_noisy_48h,vwm_quite-30h,aidatatang,aishell,thchs,magicdata,primewords,stcmds}/train || exit 1;
	utils/combine_data.sh $out_dir/data/test_combined \
                $out_dir/data/{aidatatang,aishell,magicdata,thchs}/{dev,test} || exit 1;
        utils/combine_data.sh $out_dir/data/train \
		$out_dir/data/train_combined $out_dir/data/test_combined
fi

if [ $stage -le 1 ]; then
    echo "stage 1: Generate MFCC"
  # Make MFCCs and compute the energy-based VAD for each dataset
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_xvector.conf --nj 80 --cmd "$train_cmd" \
      $out_dir/data/train $out_dir/exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $out_dir/data/train
    sid/compute_vad_decision.sh --nj 80 --cmd "$train_cmd" \
      $out_dir/data/train $out_dir/exp/make_vad $vaddir
    utils/fix_data_dir.sh $out_dir/data/train
fi

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  echo "stage 2: Augmentation"
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $out_dir/data/train/utt2num_frames > $out_dir/data/train/reco2dur

  if [ ! -d "$rirs_dir" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    echo "Download RIRS"
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_dir/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_dir/simulated_rirs/mediumroom/rir_list")

  echo "Generage reverberate"
  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    $out_dir/data/train \
    $out_dir/data/train_reverb
  cp $out_dir/data/train/vad.scp $out_dir/data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" $out_dir/data/train_reverb $out_dir/data/train_reverb.new
  rm -rf $out_dir/data/train_reverb
  mv $out_dir/data/train_reverb.new $out_dir/data/train_reverb
  ln -s $rirs_dir .

fi

if [ $stage -le 3 ]; then
  echo "stage 3: Combine Augment"
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh $out_dir/data/train_reverb 1000000 $out_dir/data/train_aug_1m
  utils/fix_data_dir.sh $out_dir/data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc_xvector.conf --nj 80 --cmd "$train_cmd" \
    $out_dir/data/train_aug_1m $out_dir/exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh $out_dir/data/train_combined $out_dir/data/train_aug_1m $out_dir/data/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  echo "stage 4: CMVN"
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/chain/xvector/prepare_feats_for_egs.sh --nj 80 --cmd "$train_cmd" \
    $out_dir/data/train_combined $out_dir/data/train_combined_no_sil $out_dir/exp/train_combined_no_sil
  utils/fix_data_dir.sh $out_dir/data/train_combined_no_sil
fi

if [ $stage -le 5 ]; then
  echo "stage 5: Clean Data"
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv $out_dir/data/train_combined_no_sil/utt2num_frames $out_dir/data/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $out_dir/data/train_combined_no_sil/utt2num_frames.bak > $out_dir/data/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl $out_dir/data/train_combined_no_sil/utt2num_frames $out_dir/data/train_combined_no_sil/utt2spk > $out_dir/data/train_combined_no_sil/utt2spk.new
  mv $out_dir/data/train_combined_no_sil/utt2spk.new $out_dir/data/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh $out_dir/data/train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' $out_dir/data/train_combined_no_sil/spk2utt > $out_dir/data/train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $out_dir/data/train_combined_no_sil/spk2num | utils/filter_scp.pl - $out_dir/data/train_combined_no_sil/spk2utt > $out_dir/data/train_combined_no_sil/spk2utt.new
  mv $out_dir/data/train_combined_no_sil/spk2utt.new $out_dir/data/train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl $out_dir/data/train_combined_no_sil/spk2utt > $out_dir/data/train_combined_no_sil/utt2spk

  utils/filter_scp.pl $out_dir/data/train_combined_no_sil/utt2spk $out_dir/data/train_combined_no_sil/utt2num_frames > $out_dir/data/train_combined_no_sil/utt2num_frames.new
  mv $out_dir/data/train_combined_no_sil/utt2num_frames.new $out_dir/data/train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $out_dir/data/train_combined_no_sil
fi

# Stages 6 through 8 are handled in run_xvector.sh
local/chain/xvector/run_xvector_1a.sh --stage $stage --train-stage 0 \
    --out_dir $out_dir
    --data $out_dir/data/train_combined_no_sil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs


if !($decode); then
	echo "Finished"
	exit 0;
fi

if [ $stage -le 9 ]; then
   echo "stage 9: extract x-vector"

   # Now we will extract x-vectors used for centering, LDA, and PLDA training.
   # Note that data/train_combined has well over 2 million utterances,
   # which is far more than is needed to train the generative PLDA model.
   # In addition, many of the utterances are very short, which causes a
   # mismatch with our evaluation conditions.  In the next command, we
   # create a data directory that contains the longest 200,000 recordings,
   # which we will use to train the backend.
   utils/subset_data_dir.sh \
     --utt-list <(sort -n -k 2 $out_dir/data/train_combined_no_sil/utt2num_frames | tail -n 200000) \
     $out_dir/data/train_combined \
     $out_dir/data/train_combined_200k

   sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
	   $nnet_dir \
	   $out_dir/data/train_combined_200k \
	   $nnet_dir/xvectors_train_combined_200k

  # Extract x-vectors used in the evaluation.
#  for name in sitw_eval_enroll sitw_eval_test sitw_dev_enroll sitw_dev_test; do
#    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
#      $nnet_dir \
#      $out_dir/data/$name \
#      $nnet_dir/xvectors_$name
#  done
fi

if [ $stage -le 10 ]; then
  # Compute the mean.vec used for centering.
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp \
    $nnet_dir/xvectors_train_combined_200k/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp ark:- |" \
    ark:$out_dir/data/train_combined_200k/utt2spk $nnet_dir/xvectors_train_combined_200k/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train_combined_200k/log/plda.log \
    ivector-compute-plda ark:$out_dir/data/train_combined_200k/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_combined_200k/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/xvectors_train_combined_200k/plda || exit 1;
fi

if [ $stage -le 11 ]; then
  # Compute PLDA scores for SITW dev core-core trials
  $train_cmd $nnet_dir/scores/log/sitw_dev_core_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnet_dir/xvectors_sitw_dev_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_combined_200k/plda - |" \
    "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:$nnet_dir/xvectors_sitw_dev_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec scp:$nnet_dir/xvectors_sitw_dev_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sitw_dev_trials_core' | cut -d\  --fields=1,2 |" $nnet_dir/scores/sitw_dev_core_scores || exit 1;

  # SITW Dev Core:
  # EER: 3.08%
  # minDCF(p-target=0.01): 0.3016
  # minDCF(p-target=0.001): 0.4993
  echo "SITW Dev Core:"
  eer=$(paste $sitw_dev_trials_core $nnet_dir/scores/sitw_dev_core_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/sitw_dev_core_scores $sitw_dev_trials_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

if [ $stage -le 12 ]; then
  # Compute PLDA scores for SITW eval core-core trials
  $train_cmd $nnet_dir/scores/log/sitw_eval_core_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnet_dir/xvectors_sitw_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_combined_200k/plda - |" \
    "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:$nnet_dir/xvectors_sitw_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec ark:- ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_combined_200k/mean.vec scp:$nnet_dir/xvectors_sitw_eval_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_combined_200k/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" $nnet_dir/scores/sitw_eval_core_scores || exit 1;

  # SITW Eval Core:
  # EER: 3.335%
  # minDCF(p-target=0.01): 0.3412
  # minDCF(p-target=0.001): 0.5106
  echo -e "\nSITW Eval Core:";
  eer=$(paste $sitw_eval_trials_core $nnet_dir/scores/sitw_eval_core_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/sitw_eval_core_scores $sitw_eval_trials_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/sitw_eval_core_scores $sitw_eval_trials_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

echo "Finished"
