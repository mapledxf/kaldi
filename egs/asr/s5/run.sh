#!/bin/bash

data=/home/data/xfding/asr-noisy-48h
trn_set=$data/wav
trans=$data/script/trans.txt
tst_set=$data/wav
out_dir=/home/data/xfding/asr-train

nj=20
stage=0
gmm_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh ${trn_set} ${trans} ${out_dir} || exit 1;
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
