#!/bin/bash

data=/home/data/xfding/train_dataset/asr
trn_set=$data/train
trans=$data/trans.txt
tst_set=$data/test
out_dir=/home/data/xfding/train_result/asr/notebook

nj=20
stage=0
gmm_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh ${trn_set} ${tst_set} ${trans} ${out_dir} || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  local/run_gmm.sh ${out_dir} --gmm_stage=$gmm_stage 
fi

# chain
if [ $stage -le 3 ]; then
  local/chain/run_tdnn.sh --nj $nj ${out_dir}
fi

local/show_results.sh

exit 0;
