#!/bin/bash

set -e -o pipefail

# This script is copied from librispeech

# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.


stage=0
train_set=train_all_cleaned # you might set this to e.g. train_all
test_sets=""
gmm=tri4a_cleaned        # This specifies a GMM-dir from the features of the type you're training the system on;
num_threads_ubm=16
num_processes=4
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.

out_dir=/home/data/xfding/train_result/asr/multi
test_enable=false

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

gmm_dir=$out_dir/exp/${gmm}
ali_dir=$out_dir/exp/${gmm}_ali_${train_set}_sp

for f in $out_dir/data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
	if [ ! -f $f ]; then
		echo "$0: expected file $f to exist"
		exit 1
	fi
done

if [ $stage -le 19 ]; then
	echo "$0: stage 19 creating chain common";
	#Although the nnet will be trained by high resolution data, we still have to
	# perturb the normal data to get the alignment.  _sp stands for speed-perturbed
	echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
	utils/data/perturb_data_dir_speed_3way.sh \
		$out_dir/data/${train_set} \
		$out_dir/data/${train_set}_sp
	echo "$0: making MFCC features for low-resolution speed-perturbed data"
	steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 50 \
		$out_dir/data/${train_set}_sp || exit 1;
	steps/compute_cmvn_stats.sh \
		$out_dir/data/${train_set}_sp || exit 1;
  	echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  	echo ".. speed-perturbed segments were too short."
  	utils/fix_data_dir.sh \
		$out_dir/data/${train_set}_sp
fi

if [ $stage -le 20 ]; then
	echo "$0: stage 20 align fmllr";
	if [ -f $ali_dir/ali.1.gz ]; then
		echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
		echo " ... or use a later --stage option."
		exit 1
	fi
	echo "$0: aligning with the perturbed low-resolution data"
	steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
		$out_dir/data/${train_set}_sp \
		$out_dir/data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 21 ]; then
	# Create high-resolution MFCC features (with 40 cepstra instead of 13).
	# this shows how you can split across multiple file-systems.  we'll split the
	# MFCC dir across multiple locations.  You might want to be careful here, if you
	# have multiple copies of Kaldi checked out and run the same recipe, not to let
	# them overwrite each other.
	echo "$0: stage 21 creating high-resolution MFCC features"
	mfccdir=$out_dir/data/${train_set}_sp_hires/data

	utils/copy_data_dir.sh \
		$out_dir/data/${train_set}_sp \
		$out_dir/data/${train_set}_sp_hires
        if $test_enable; then
		for datadir in $test_sets; do
    			utils/copy_data_dir.sh \
				$out_dir/data/$datadir/test \
				$out_dir/data/$datadir/test_hires
  		done
	fi

	# do volume-perturbation on the training data prior to extracting hires
	# features; this helps make trained nnets more invariant to test data volume.
	utils/data/perturb_data_dir_volume.sh \
		$out_dir/data/${train_set}_sp_hires
	steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
		--cmd "$train_cmd" \
		$out_dir/data/${train_set}_sp_hires || exit 1;
	steps/compute_cmvn_stats.sh \
		$out_dir/data/${train_set}_sp_hires || exit 1;
	utils/fix_data_dir.sh \
		$out_dir/data/${train_set}_sp_hires

	if $test_enable; then
		for datadir in $test_sets; do
			steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
				--cmd "$train_cmd" \
				$out_dir/data/$datadir/test_hires || exit 1;
    			steps/compute_cmvn_stats.sh \
				$out_dir/data/$datadir/test_hires || exit 1;
    			utils/fix_data_dir.sh \
				$out_dir/data/$datadir/test_hires
		done
	fi

	# now create a data subset.  60k is 1/5th of the training dataset (around 200 hours).
	utils/subset_data_dir.sh \
		$out_dir/data/${train_set}_sp_hires 60000 \
    		$out_dir/data/${train_set}_sp_hires_60k
fi


if [ $stage -le 22 ]; then
	echo "$0: stage 22 making a subset of data to train the diagonal UBM and the PCA transform."
	# We'll use one hundredth of the data, since whole training set is very large.
	mkdir -p $out_dir/exp/nnet3${nnet3_affix}/diag_ubm
	temp_data_root=$out_dir/exp/nnet3${nnet3_affix}/diag_ubm

	num_utts_total=$(wc -l <${out_dir}/data/${train_set}_sp_hires/utt2spk)
	num_utts=$[$num_utts_total/100]
	utils/data/subset_data_dir.sh \
		$out_dir/data/${train_set}_sp_hires \
		$num_utts \
		${temp_data_root}/${train_set}_sp_hires_subset

	echo "$0: computing a PCA transform from the hires data."
	steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
		--splice-opts "--left-context=3 --right-context=3" \
		--max-utts 10000 --subsample 2 \
		${temp_data_root}/${train_set}_sp_hires_subset \
		$out_dir/exp/nnet3${nnet3_affix}/pca_transform

	echo "$0: training the diagonal UBM."
	# Use 512 Gaussians in the UBM.
	steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
		--num-frames 700000 \
		--num-threads $num_threads_ubm \
		${temp_data_root}/${train_set}_sp_hires_subset 512 \
		$out_dir/exp/nnet3${nnet3_affix}/pca_transform \
		$out_dir/exp/nnet3${nnet3_affix}/diag_ubm
fi


if [ $stage -le 23 ]; then
	# iVector extractors can in general be sensitive to the amount of data, but
	# this one has a fairly small dim (defaults to 100) so we don't use all of it,
	# we use just the 60k subset (about one fifth of the data, or 200 hours).
	echo "$0: stage 23 training the iVector extractor"
	steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
		--num-processes $num_processes \
		$out_dir/data/${train_set}_sp_hires_60k \
		$out_dir/exp/nnet3${nnet3_affix}/diag_ubm \
		$out_dir/exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 24 ]; then
	echo "$0: stage 24 extracting iVectors for training data"
	ivectordir=$out_dir/exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
	# We extract iVectors on the speed-perturbed training data after combining
	# short segments, which will be what we train the system on.  With
	# --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
	# each of these pairs as one speaker. this gives more diversity in iVectors..
	# Note that these are extracted 'online'.

	# having a larger number of speakers is helpful for generalization, and to
	# handle per-utterance decoding well (iVector starts at zero).
	utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
		$out_dir/data/${train_set}_sp_hires \
		${ivectordir}/${train_set}_sp_hires_max2

	steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
		${ivectordir}/${train_set}_sp_hires_max2 \
		$out_dir/exp/nnet3${nnet3_affix}/extractor \
		$ivectordir || exit 1;
fi

if [ $stage -le 25 ]; then
        echo "$0: stage 25 extracting iVectors for test data"
        if $test_enable; then
		for data in $test_sets; do
			steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
				$out_dir/data/${data}/test_hires \
				$out_dir/exp/nnet3${nnet3_affix}/extractor \
				$out_dir/exp/nnet3${nnet3_affix}/ivectors_${data}_hires || exit 1;
  		done
	fi
fi
echo "$0: iVectors extracted successfully"
