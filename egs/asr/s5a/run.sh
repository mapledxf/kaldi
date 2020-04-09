#!/bin/bash

# Copyright 2019 Microsoft Corporation (authors: Xingyu Na)
# Apache 2.0

. ./cmd.sh
. ./path.sh

stage=0

test_sets="aishell aidatatang magicdata thchs"
corpus_lm=false   # interpolate with corpus lm

openslr_aidatatang=/home/zlj/Datasets/aidatatang_200zh
openslr_aishell=/home/data/xfding/dataset/asr/aishell/data_aishell
openslr_magicdata=/home/data/xfding/dataset/asr/magicdata
openslr_primewords=/home/data/xfding/dataset/asr/primewords_md_2018_set1
openslr_stcmds=/home/data/xfding/dataset/asr/ST-CMDS-20170001_1-OS
openslr_thchs=/home/data/xfding/dataset/asr/thchs30/data_thchs30

vwm_noisy_48h_src=/home/zlj/dxf/ASR/VWM/asr-noisy-48h
vwm_quite_30h_src=/home/zlj/dxf/ASR/VWM/asr-quite-30h
vwm_noisy_48h_out=/home/zlj/dxf/train_dataset/asr/noisy-48h
vwm_quite_30h_out=/home/zlj/dxf/train_dataset/asr/quite-30h

out_dir=/home/zlj/dxf/train_result/asr/multi

#test for result
test_enable=false

. utils/parse_options.sh

#Data preparation
if [ $stage -le 1 ]; then
        local/vwm_data_prep.sh $vwm_noisy_48h_src $vwm_noisy_48h_out $out_dir/data/vwm_noisy_48h || exit 1;
        local/vwm_data_prep.sh $vwm_quite_30h_src $vwm_quite_30h_out $out_dir/data/vwm_quite-30h || exit 1;

	local/aidatatang_data_prep.sh $openslr_aidatatang $out_dir/data/aidatatang || exit 1;
#	local/aishell_data_prep.sh $openslr_aishell $out_dir/data/aishell || exit 1;
#	local/thchs-30_data_prep.sh $openslr_thchs $out_dir/data/thchs || exit 1;
#	local/magicdata_data_prep.sh $openslr_magicdata $out_dir/data/magicdata || exit 1;
#	local/primewords_data_prep.sh $openslr_primewords $out_dir/data/primewords || exit 1;
#	local/stcmds_data_prep.sh $openslr_stcmds $out_dir/data/stcmds || exit 1;
fi

echo "$0: stage 1 data preparation completed"

#Dictionary generation
if [ $stage -le 2 ]; then
	# normalize transcripts
	utils/combine_data.sh $out_dir/data/train_combined \
                $out_dir/data/{vwm_noisy_48h,vwm_quite-30h,aidatatang}/train || exit 1;
#               $out_dir/data/{vwm_noisy_48h,vwm_quite-30h,aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train || exit 1;
	utils/combine_data.sh $out_dir/data/test_combined \
                $out_dir/data/aidatatang/{dev,test} || exit 1;
#               $out_dir/data/{aidatatang,aishell,magicdata,thchs}/{dev,test} || exit 1;
	local/prepare_dict.sh --out_dir $out_dir || exit 1;
fi

echo "$0: stage 2 dictionary generation completed"

#LM preparation
if [ $stage -le 3 ]; then
	# train LM using transcription
	local/train_lms.sh --out_dir $out_dir || exit 1;
fi

echo "$0: stage 3 LM preparation completed"

#LM generation
if [ $stage -le 4 ]; then
	# prepare LM
	utils/prepare_lang.sh \
		$out_dir/data/local/dict \
		"<UNK>" \
		$out_dir/data/local/lang \
		$out_dir/data/lang || exit 1;
	utils/format_lm.sh \
		$out_dir/data/lang \
		$out_dir/data/local/lm/3gram-mincount/lm_unpruned.gz \
		$out_dir/data/local/dict/lexicon.txt \
		$out_dir/data/lang_combined_tg || exit 1;
fi

echo "$0: stage 4 LM generation completed"

#MFCC generation for train set
if [ $stage -le 5 ]; then
	# make features
	mfccdir=mfcc
#	corpora="vwm_noisy_48 aidatatang aishell magicdata primewords stcmds thchs"
        corpora="vwm_noisy_48h vwm_quite-30h aidatatang"
	for c in $corpora; do
	(
		steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 20 \
			$out_dir/data/$c/train \
			$out_dir/exp/make_mfcc/$c/train \
			$mfccdir/$c || exit 1;
		steps/compute_cmvn_stats.sh \
			$out_dir/data/$c/train \
			$out_dir/exp/make_mfcc/$c/train \
			$mfccdir/$c || exit 1;
	) &
	done
	wait
fi

echo "$0: stage 5 MFCC generation for train set completed"

#MFCC generation for test set
if [ $stage -le 6 ]; then
        if $test_enable; then
		# make test features
		mfccdir=mfcc
		for c in $test_sets; do
		(
			steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 10 \
				$out_dir/data/$c/test \
				$out_dir/exp/make_mfcc/$c/test \
				$mfccdir/$c || exit 1;
			steps/compute_cmvn_stats.sh \
				$out_dir/data/$c/test \
				exp/make_mfcc/$c/test \
				$mfccdir/$c || exit 1;
		) &
		done
		wait
	fi
fi

echo "$0: stage 6 MFCC generation for test set completed"

#Train mono
if [ $stage -le 7 ]; then
	# train mono and tri1a using aishell(~120k)
	# mono has been used in aishell recipe, so no test
	utils/combine_data.sh \
		$out_dir/data/train_mono \
		$out_dir/data/{vwm_quite-30h,vwm_noisy_48h,aidatatang}/train || exit 1;

	steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
		$out_dir/data/train_mono \
		$out_dir/data/lang \
		$out_dir/exp/mono || exit 1;
	steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
		$out_dir/data/train_mono \
		$out_dir/data/lang \
		$out_dir/exp/mono \
		$out_dir/exp/mono_ali || exit 1;
	steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 20000 \
		$out_dir/data/train_mono \
		$out_dir/data/lang \
		$out_dir/exp/mono_ali \
		$out_dir/exp/tri1a || exit 1;
fi

echo "$0: stage 7 train mono completed"

#Train tri1b
if [ $stage -le 8 ]; then
	# train tri1b using vwm + aishell + primewords + stcmds + thchs (~280k)
	utils/combine_data.sh \
		$out_dir/data/train_280k \
                $out_dir/data/{vwm_quite-30h,vwm_noisy_48h,aidatatang}/train || exit 1;
#               $out_dir/data/{vwm_quite-30h,vwm_noisy_48h,aidatatang,primewords,stcmds,thchs}/train || exit 1;

	steps/align_si.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" \
		$out_dir/data/train_280k \
		$out_dir/data/lang \
		$out_dir/exp/tri1a \
		$out_dir/exp/tri1a_280k_ali || exit 1;
	steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 4500 36000 \
		$out_dir/data/train_280k \
		$out_dir/data/lang \
		$out_dir/exp/tri1a_280k_ali \
		$out_dir/exp/tri1b || exit 1;
fi

echo "$0: stage 8 train tri1b completed"

#Test tri1b
if [ $stage -le 9 ]; then
	# test tri1b
	utils/mkgraph.sh \
		$out_dir/data/lang_combined_tg \
		$out_dir/exp/tri1b \
		$out_dir/exp/tri1b/graph_tg || exit 1;
        if $test_enable; then
		for c in $test_sets; do
		(
			steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
				$out_dir/exp/tri1b/graph_tg \
				$out_dir/data/$c/test \
				$out_dir/exp/tri1b/decode_${c}_test_tg || exit 1;
		) &
		done
		wait
	fi
fi

echo "$0: stage 9 test tri1b completed"

#Train tri2a
if [ $stage -le 10 ]; then
	# train tri2a using train_280k
	steps/align_si.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" \
		$out_dir/data/train_280k \
		$out_dir/data/lang \
		$out_dir/exp/tri1b \
		$out_dir/exp/tri1b_280k_ali || exit 1;
	steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 5500 90000 \
		$out_dir/data/train_280k \
		$out_dir/data/lang \
		$out_dir/exp/tri1b_280k_ali \
		$out_dir/exp/tri2a || exit 1;
fi

echo "$0: stage 10 train tri2a completed"

#Test tri2a
if [ $stage -le 11 ]; then
	# test tri2a
	utils/mkgraph.sh \
		$out_dir/data/lang_combined_tg \
		$out_dir/exp/tri2a \
		$out_dir/exp/tri2a/graph_tg || exit 1;
	if $test_enable; then
		for c in $test_sets; do
		(
			steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
				$out_dir/exp/tri2a/graph_tg \
				$out_dir/data/$c/test \
				$out_dir/exp/tri2a/decode_${c}_test_tg || exit 1;
		) &
		done
		wait
	fi
fi

echo "$0: stage 11 test tri2a completed"

#Train tri3a
if [ $stage -le 12 ]; then
	# train tri3a using aidatatang + aishell + primewords + stcmds + thchs (~440k)
	utils/combine_data.sh \
		$out_dir/data/train_440k \
                $out_dir/data/{vwm_quite-30h,vwm_noisy_48h,aidatatang}/train || exit 1;
#               $out_dir/data/{vwm_quite-30h,vwm_noisy_48h,aidatatang,aishell,primewords,stcmds,thchs}/train || exit 1;

	steps/align_si.sh --boost-silence 1.25 --nj 60 --cmd "$train_cmd" \
		$out_dir/data/train_440k \
		$out_dir/data/lang \
		$out_dir/exp/tri2a \
		$out_dir/exp/tri2a_440k_ali || exit 1;
	steps/train_lda_mllt.sh --cmd "$train_cmd" 7000 110000 \
		$out_dir/data/train_440k \
		$out_dir/data/lang \
		$out_dir/exp/tri2a_440k_ali \
		$out_dir/exp/tri3a || exit 1;
fi

echo "$0: stage 12 train tri3a completed"

#Test tri3a
if [ $stage -le 13 ]; then
	# test tri3a
	utils/mkgraph.sh \
		$out_dir/data/lang_combined_tg \
		$out_dir/exp/tri3a \
		$out_dir/exp/tri3a/graph_tg || exit 1;
        if $test_enable; then
      		for c in $test_sets; do
		(
			steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
				$out_dir/exp/tri3a/graph_tg \
				$out_dir/data/$c/test \
				$out_dir/exp/tri3a/decode_${c}_test_tg || exit 1;
		) &
		done
		wait
	fi
fi

echo "$0: stage 13 test tri3a completed"

#Train tri4a
if [ $stage -le 14 ]; then
	# train tri4a using all
	utils/combine_data.sh \
		$out_dir/data/train_all \
                $out_dir/data/{vwm_quite-30h,vwm_noisy_48h,aidatatang}/train || exit 1;
#               $out_dir/data/{vwm_quite-30h,vwm_noisy_48h,aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train || exit 1;

	steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
		$out_dir/data/train_all \
		$out_dir/data/lang \
		$out_dir/exp/tri3a \
		$out_dir/exp/tri3a_ali || exit 1;
	steps/train_sat.sh --cmd "$train_cmd" 12000 190000 \
		$out_dir/data/train_all \
		$out_dir/data/lang \
		$out_dir/exp/tri3a_ali \
		$out_dir/exp/tri4a || exit 1;
fi

echo "$0: stage 14 train tri4a completed"

#Test tri4a
if [ $stage -le 15 ]; then
	# test tri4a
	utils/mkgraph.sh \
		$out_dir/data/lang_combined_tg \
		$out_dir/exp/tri4a \
		$out_dir/exp/tri4a/graph_tg || exit 1;
        if $test_enable; then
		for c in $test_sets; do
		(
			steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
				$out_dir/exp/tri4a/graph_tg \
				$out_dir/data/$c/test \
				$out_dir/exp/tri4a/decode_${c}_test_tg || exit 1;
		) &
		done
		wait
	fi
fi

echo "$0: stage 15 test tri4a completed"

#Clean up
if [ $stage -le 16 ]; then
	# run clean and retrain
	local/run_cleanup_segmentation.sh \
		--nj 10 \
		--test-sets "$test_sets" \
		--test-enable "$test_enable" \
		--out-dir "$out_dir" || exit 1;
fi

echo "$0: stage 16 clean up completed"

#Collect WER
if [ $stage -le 17 ]; then
        if $test_enable; then
		# collect GMM test results
		for c in $test_sets; do
			echo "$c test set results"
			for x in $out_dir/exp/*/decode_${c}*_tg; do
				grep WER $x/cer_* | utils/best_wer.sh
			done
			echo ""
		done
	fi
fi

echo "$0: stage 17 collect WER completed"

#Train chain. The following steps are handled in this script.
# chain modeling script
local/chain/run_cnn_tdnn.sh \
    --out_dir $out_dir \
	--stage $stage \
	--test-sets "$test_sets" \
	--test-enable "$test_enable"
if $test_enable; then
	for c in $test_sets; do
		for x in $out_dir/exp/chain_cleaned/*/decode_${c}*_tg; do
			grep WER $x/cer_* | utils/best_wer.sh
		done
	done
fi

echo "$0: Training complete"
