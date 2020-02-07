#!/bin/bash

base_data=/home/data/xfding/train_dataset/asr/aidatatang_200zh
base_train=$base_data/corpus/train
base_test=$base_data/corpus/test
base_script=$base_data/transcript/aidatatang_200_zh_transcript.txt

extra_data=/home/data/xfding/train_dataset/asr/ebo
extra_train=$extra_data/train
extra_test=$extra_data/test
extra_script=$extra_data/trans.txt

out_dir=/home/data/xfding/train_result/asr/notebook

nj=20
stage=0
gmm_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  echo 'Prepare all'
  local/prepare_all.sh \
	  ${base_train} ${base_test} ${base_script} \
	  ${extra_train} ${extra_test} ${extra_script} \
	  ${out_dir} || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  echo 'Prepare GMM'
  local/run_gmm.sh ${out_dir} --gmm_stage=$gmm_stage 
fi

# chain
if [ $stage -le 3 ]; then
  echo 'Train TDNN'
  local/chain/run_tdnn.sh --nj $nj ${out_dir}
fi

local/show_results.sh ${out_dir}

exit 0;
