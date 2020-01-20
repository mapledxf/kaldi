#!/bin/bash

out_dir=/home/data/xfding/asr-train

./utils/mkgraph.sh --self-loop-scale 1.0 \
	$out_dir/data/lang_chain \
	$out_dir/exp/chain/tdnn_1b_all_sp \
       	$out_dir/exp/chain/tdnn_1b_all_sp/graph

./steps/online/nnet3/prepare_online_decoding.sh \
	--add_pitch true \
	$out_dir/data/lang_chain \
	$out_dir/exp/chain/extractor_all \
	$out_dir/exp/chain/tdnn_1b_all_sp \
	/opt/models/chinese/vwm

cp $out_dir/exp/chain/tdnn_1b_all_sp/graph/HCLG.fst \
	$out_dir/exp/chain/tdnn_1b_all_sp/graph/words.txt \
	/opt/models/chinese/vwm/
