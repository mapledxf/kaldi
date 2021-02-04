#!/bin/bash

# Copyright 2019 Microsoft Corporation (authors: Xingyu Na)
# Apache 2.0

. ./cmd.sh
. ./path.sh

stage=0

fs=16000

nj=64

out_dir=/data/xfding/train_result/asr/vwm
data_dir=/data/xfding/data_prep/asr
cmudict=/data/xfding/pretrained_model/dict/cmudict
cedict=/data/xfding/pretrained_model/dict/cedict

#test for result
test_enable=false

. utils/parse_options.sh

test_sets="aishell aidatatang magicdata thchs"
corpus="vwm_noisy_48h vwm_quite_30h cmlr csmsc aidatatang aishell magicdata primewords stcmds thchs"

#Data preparation
if [ $stage -le 1 ]; then
	/home/xfding/vwm_data_prepare/data_prep.sh --is-tts false --fs $fs aidatatang,aishell,cmlr,csmsc,vwm_noisy_48h,vwm_quite_30h,thchs,magicdata,primewords,stcmds
fi
echo -e "$0: stage 1 data preparation completed\n"

#Dictionary generation
if [ $stage -le 2 ]; then
	utils/combine_data.sh $out_dir/data/train_all \
                $data_dir/{aidatatang,aishell,cmlr,csmsc,vwm_noisy_48h,vwm_quite_30h,thchs,magicdata,primewords,stcmds}_${fs}/{train,dev} || exit 1;
	utils/combine_data.sh $out_dir/data/test_all \
                $data_dir/{aishell,aidatatang,magicdata,thchs}_${fs}/test || exit 1;
	local/prepare_dict.sh --out-dir $out_dir --cmudict $cmudict --cedict $cedict || exit 1;
fi
echo -e "$0: stage 2 dictionary generation completed\n"

#LM preparation
if [ $stage -le 3 ]; then
	# train LM using transcription
	local/train_lms.sh --out-dir $out_dir || exit 1;
fi
echo -e "$0: stage 3 LM preparation completed\n"

#LM generation
if [ $stage -le 4 ]; then
	# prepare LM
	utils/prepare_lang.sh \
		$out_dir/data/local/dict_nosp \
		"<UNK>" \
		$out_dir/data/local/lang_tmp_nosp \
		$out_dir/data/lang_nosp || exit 1;
	
	if $test_enable; then
		utils/format_lm.sh \
			$out_dir/data/lang_nosp \
			$out_dir/data/local/lm/3gram-mincount/lm_unpruned.gz \
			$out_dir/data/local/dict_nosp/lexicon.txt \
			$out_dir/data/lang_combined_tg || exit 1;
	fi
fi
echo -e "$0: stage 4 LM generation completed\n"

mfccdir=$out_dir/mfcc
#MFCC generation for train set
if [ $stage -le 5 ]; then
	# make features
	for c in $corpus; do
	(
		utils/fix_data_dir.sh \
			$data_dir/${c}_${fs}/train
		steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
			$data_dir/${c}_${fs}/train \
			$out_dir/exp/make_mfcc/$c/train \
			$mfccdir/$c || exit 1;
		steps/compute_cmvn_stats.sh \
                        $data_dir/${c}_${fs}/train \
			$out_dir/exp/make_mfcc/$c/train \
			$mfccdir/$c || exit 1;
	) &
	done
	wait
	# used for fsmn training
	utils/fix_data_dir.sh \
		$out_dir/data/test_all
	steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
		$out_dir/data/test_all \
		$out_dir/exp/make_mfcc/test_all \
		$mfccdir/test_all || exit 1;
	steps/compute_cmvn_stats.sh \
		$out_dir/data/test_all \
		$out_dir/exp/make_mfcc/test_all \
		$mfccdir/test_all || exit 1;
fi
echo -e "$0: stage 5 MFCC generation for train set and test set completed\n"

#MFCC generation for test set
if [ $stage -le 6 ]; then
	if $test_enable; then
		# make test features
		for c in $test_sets; do
		(
			utils/fix_data_dir.sh \
				$data_dir/${c}_${fs}/test
			steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
				$data_dir/${c}_${fs}/test \
				$out_dir/exp/make_mfcc/$c/test \
				$mfccdir/$c || exit 1;
			steps/compute_cmvn_stats.sh \
				$data_dir/${c}_${fs}/test \
				$out_dir/exp/make_mfcc/$c/test \
				$mfccdir/$c || exit 1;
		) &
		done
#		wait
	fi
fi
echo -e "$0: stage 6 MFCC generation for test set completed\n"

#Train mono 单因素训练
if [ $stage -le 7 ]; then
	# train mono and tri1a using aishell(~120k)
	# mono has been used in aishell recipe, so no test
	utils/subset_data_dir.sh --shortest $data_dir/aishell_${fs}/train 2000 $out_dir/data/train_mono_small
	utils/subset_data_dir.sh $data_dir/aishell_${fs}/train 10000 $out_dir/data/train_mono_large

	steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
		$out_dir/data/train_mono_small \
		$out_dir/data/lang_nosp \
		$out_dir/exp/mono || exit 1;
	steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
		$out_dir/data/train_mono_large \
		$out_dir/data/lang_nosp \
		$out_dir/exp/mono \
		$out_dir/exp/mono_ali || exit 1;
	steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 20000 \
		$out_dir/data/train_mono_large \
		$out_dir/data/lang_nosp \
		$out_dir/exp/mono_ali \
		$out_dir/exp/tri1a || exit 1;
fi
echo -e "$0: stage 7 train mono completed\n"

#Train tri1b 非说话人自适应，mllt的作用是减少协方差矩阵对角化的损失
if [ $stage -le 8 ]; then
	# train tri1b using aishell + aidatatang
	utils/combine_data.sh \
		$out_dir/data/train_small \
                $data_dir/{aidatatang,aishell}_${fs}/train || exit 1;

	steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		$out_dir/data/train_small \
		$out_dir/data/lang_nosp \
		$out_dir/exp/tri1a \
		$out_dir/exp/tri1a_ali || exit 1;
	steps/train_lda_mllt.sh --cmd "$train_cmd" 2500 15000 \
		$out_dir/data/train_small \
		$out_dir/data/lang_nosp \
		$out_dir/exp/tri1a_ali \
		$out_dir/exp/tri1b || exit 1;
fi
echo -e "$0: stage 8 train tri1b completed\n"

#Test tri1b
if [ $stage -le 9 ]; then
	# test tri1b
	if $test_enable; then
		utils/mkgraph.sh \
			$out_dir/data/lang_combined_tg \
			$out_dir/exp/tri1b \
			$out_dir/exp/tri1b/graph_tg || exit 1;
		for c in $test_sets; do
		(
			steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $nj \
				$out_dir/exp/tri1b/graph_tg \
				$data_dir/$c_${fs}/test \
				$out_dir/exp/tri1b/decode_${c}_test_tg || exit 1;
		) &
		done
#		wait
	fi
fi
echo -e "$0: stage 9 test tri1b completed\n"

#Train tri2a 说话人自适应模型 fmllr训练脚本
if [ $stage -le 10 ]; then
	# train tri2a using train_small
	steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		$out_dir/data/train_small \
		$out_dir/data/lang_nosp \
		$out_dir/exp/tri1b \
		$out_dir/exp/tri1b_ali || exit 1;
	steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
		$out_dir/data/train_small \
		$out_dir/data/lang_nosp \
		$out_dir/exp/tri1b_ali \
		$out_dir/exp/tri2a || exit 1;
fi
echo -e "$0: stage 10 train tri2a completed\n"

#Test tri2a
if [ $stage -le 11 ]; then
	# test tri2a
	if $test_enable; then
		utils/mkgraph.sh \
			$out_dir/data/lang_combined_tg \
			$out_dir/exp/tri2a \
			$out_dir/exp/tri2a/graph_tg || exit 1;
		for c in $test_sets; do
		(
			steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $nj \
				$out_dir/exp/tri2a/graph_tg \
				$data_dir/$c_${fs}/test \
				$out_dir/exp/tri2a/decode_${c}_test_tg || exit 1;
		) &
		done
#		wait
	fi
fi
echo -e "$0: stage 11 test tri2a completed\n"

#Train tri3a
if [ $stage -le 12 ]; then
	# train tri3a using aidatatang + aishell + csmsc
	utils/combine_data.sh \
		$out_dir/data/train_medium \
				$out_dir/data/train_small \
                $data_dir/csmsc_${fs}/train || exit 1;
	steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
		$out_dir/data/train_medium \
		$out_dir/data/lang_nosp \
		$out_dir/exp/tri2a \
		$out_dir/exp/tri2a_ali || exit 1;
	steps/train_sat.sh --cmd "$train_cmd" 7000 110000 \
		$out_dir/data/train_medium \
		$out_dir/data/lang_nosp \
		$out_dir/exp/tri2a_ali \
		$out_dir/exp/tri3a || exit 1;
fi
echo -e "$0: stage 12 train tri3a completed\n"


#Test tri3a
if [ $stage -le 13 ]; then
	# test tri3a
	if $test_enable; then
		utils/mkgraph.sh \
			$out_dir/data/lang_combined_tg \
			$out_dir/exp/tri3a \
			$out_dir/exp/tri3a/graph_tg || exit 1;
		for c in $test_sets; do
		(
			steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj $nj \
				$out_dir/exp/tri3a/graph_tg \
				$data_dir/$c_${fs}/test \
				$out_dir/exp/tri3a/decode_${c}_test_tg || exit 1;
		) &
		done
#		wait
	fi
fi
echo -e "$0: stage 13 test tri3a completed\n"

#Re-compute LM
if [ $stage -le 14 ]; then
	steps/get_prons.sh --cmd "$train_cmd" \
		$out_dir/data/train_medium \
		$out_dir/data/lang_nosp \
		$out_dir/exp/tri3a || exit 1;
	utils/dict_dir_add_pronprobs.sh --max-normalize true \
		$out_dir/data/local/dict_nosp \
		$out_dir/exp/tri3a/pron_counts_nowb.txt \
		$out_dir/exp/tri3a/sil_counts_nowb.txt \
		$out_dir/exp/tri3a/pron_bigram_counts_nowb.txt \
		$out_dir/data/local/dict
	utils/prepare_lang.sh \
		$out_dir/data/local/dict \
		"<UNK>" \
		$out_dir/data/local/lang_tmp \
		$out_dir/data/lang || exit 1;
		
	if $test_enable; then
		utils/format_lm.sh \
			$out_dir/data/lang \
			$out_dir/data/local/lm/3gram-mincount/lm_unpruned.gz \
			$out_dir/data/local/dict/lexicon.txt \
			$out_dir/data/lang_combined_tg || exit 1;
	fi
fi
echo -e "$0: stage 14 re-compute lm completed\n"

#run NN base one medium dataset
if [ $stage -le 15 ]; then
	steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
		$out_dir/data/train_medium \
		$out_dir/data/lang \
		$out_dir/exp/tri3a \
		$out_dir/exp/tri3a_ali || exit 1;
	local/nnet/run_3a.sh --out_dir $out_dir
fi
echo -e "$0: stage 15 NN on medium dataset  completed\n"

#Train tri4a
if [ $stage -le 16 ]; then
	# train tri4a using aishell + aidatatang + csmsc + cmlr
	utils/combine_data.sh \
		$out_dir/data/train_large \
		$out_dir/data/train_medium \
                $data_dir/cmlr_${fs}/train || exit 1;

	steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
		$out_dir/data/train_large \
		$out_dir/data/lang \
		$out_dir/exp/tri3a \
		$out_dir/exp/tri3a_ali || exit 1;
	steps/train_sat.sh --cmd "$train_cmd" 5000 100000 \
		$out_dir/data/train_large \
		$out_dir/data/lang \
		$out_dir/exp/tri3a_ali \
		$out_dir/exp/tri4a || exit 1;
fi
echo -e "$0: stage 16 train tri4a completed\n"

#Test tri4a
if [ $stage -le 17 ]; then
	# test tri4a
	if $test_enable; then
     		utils/mkgraph.sh \
			$out_dir/data/lang_combined_tg \
			$out_dir/exp/tri4a \
			$out_dir/exp/tri4a/graph_tg || exit 1;
		for c in $test_sets; do
		(
			steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj $nj \
				$out_dir/exp/tri4a/graph_tg \
				$data_dir/$c_${fs}/test \
				$out_dir/exp/tri4a/decode_${c}_test_tg || exit 1;
		) &
		done
#		wait
	fi
fi
echo -e "$0: stage 17 test tri4a completed\n"

#Run NN base on large dataset
if [ $stage -le 18 ]; then
	local/nnet/run_4a.sh --out_dir $out_dir
fi
echo -e "$0: stage 18 NN on large dataset completed\n"

#Train tri5a
if [ $stage -le 19 ]; then
	# train tri5a using all
	utils/combine_data.sh \
		$out_dir/data/train_full \
				$out_dir/data/train_large \
                $data_dir/{thchs,magicdata,primewords,stcmds}_${fs}/train || exit 1;

	steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
		$out_dir/data/train_full \
		$out_dir/data/lang \
		$out_dir/exp/tri4a \
		$out_dir/exp/tri4a_ali || exit 1;
	steps/train_quick.sh --cmd "$train_cmd" 7000 150000 \
		$out_dir/data/train_full \
		$out_dir/data/lang \
		$out_dir/exp/tri4a_ali \
		$out_dir/exp/tri5a || exit 1;
fi
echo -e "$0: stage 19 train tri5a completed\n"

#Clean and segment
if [ $stage -le 20 ]; then
	local/run_cleanup_segmentation.sh \
		--nj $nj \
		--data $out_dir/data/train_full \
		--srcdir $out_dir/exp/tri5a \
		--test-sets "$test_sets" \
		--test-enable "$test_enable" \
		--out-dir $out_dir || exit 1;
fi
echo -e "$0: stage 20 clean and segment completed\n"


#Collect WER
if [ $stage -le 21 ]; then
    if $test_enable; then
		# collect GMM test results
		for c in $test_sets; do
		(
			echo "$c test set results"
			for x in $out_dir/exp/*/decode_${c}*_tg; do
				grep WER $x/cer_* | utils/best_wer.sh
			done
			echo ""
		) &
		done
#		wait
	fi
fi
echo -e "$0: stage 21 collect WER completed\n"

#Train chain. The following steps are handled in this script.
# chain modeling script
# local/chain/run_cnn_tdnn.sh \
# 	--stage $stage \
# 	--out-dir $out_dir \
# 	--test-sets "$test_sets" \
# 	--test-enable "$test_enable"
# if $test_enable; then
# 	for c in $test_sets; do
# 		for x in $out_dir/exp/chain_cleaned/*/decode_${c}*_tg; do
# 			grep WER $x/cer_* | utils/best_wer.sh
# 		done
# 	done
# fi

# Train ivector
if [ $stage -le 22 ]; then
    min_seg_len=1.55
    nnet3_affix=_cleaned
    local/nnet/run_ivector_common.sh --stage 0 \
                                   --min-seg-len $min_seg_len \
                                   --out-dir $out_dir \
                                   --train-set train_full_cleaned \
                                   --gmm_dir $out_dir/exp/tri5a_cleaned \
                                   --num-threads-ubm 6 --num-processes 3 \
                                   --nnet3-affix "$nnet3_affix" || exit 1;
fi
echo -e "$0: stage 22 ivector extraced\n"

## Traing FSMN models on the cleaned-up data
# if [ $stage -le 23 ]; then
## Three configurations of DFSMN with different model size (S--small; M--medium; L--large)
## with/without online ivector

#local/nnet/run_fsmn.sh \
#	--stage 0 \
#	--gmm_dir $out_dir/exp/tri5a_cleaned \
#	--out-dir $out_dir \
#	--train train_full_cleaned \
#	--dev test_all \
#	DFSMN_S

# local/nnet/run_fsmn.sh DFSMN_M
# local/nnet/run_fsmn.sh DFSMN_L
# local/nnet/run_fsmn_ivector.sh DFSMN_S_ivector
# local/nnet/run_fsmn_ivector.sh DFSMN_M_ivector
# local/nnet/run_fsmn_ivector.sh DFSMN_L_ivector
# fi
echo -e "$0: stage 23 train fsmn done\n"


if [ $stage -le 24 ]; then
	#Train chain. The following steps are handled in this script.
	# chain modeling script
	local/chain/tuning/run_tdnn_1j.sh \
		--stage 0 \
	 	--out-dir $out_dir \
	 	--test-sets "$test_sets" \
		--train_set train_full_cleaned \
		--gmm tri5a_cleaned 
	if $test_enable; then
		for c in $test_sets; do
			for x in $out_dir/exp/chain_cleaned/*/decode_${c}*_tg; do
				grep WER $x/cer_* | utils/best_wer.sh
			done
		done
	fi
fi
echo -e "$0: stage 24 train tdnn done\n"

#if [ $stage -le 25 ]; then
#	./local/lookahead/run_lookahead.sh \
#		--out-dir $out_dir \
#		--lm ${out_dir}/data/local/lm/3gram-mincount/lm_unpruned.gz \
#		--am ${out_dir}/exp/chain_cleaned/tdnn1j_sp
#fi

echo -e "$0: Training complete\n"
