#!/bin/bash

stage=-1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "prepare_all.sh <corpus-train-dir> <test_dir> <corpus-train-transcript> <output-dir>"
  echo " e.g prepare_all.sh /data/train /data/test /data/train/trans.txt /data/output"
  exit 1;
fi

trn_set=$1
test_set=$2
trn_trans=$3
out_dir=$4

echo "transcript ${trn_trans}"
echo "train set ${trn_set}"
echo "test set ${test_set}"
echo "out dir ${out_dir}"

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 1 ]; then
  local/prepare_dict.sh ${out_dir}/data/local/dict || exit 1;
fi

if [ $stage -le 2 ]; then
  local/prepare_data.sh ${trn_set} ${trn_trans} $out_dir/data/local/dict $out_dir/data/local/train $out_dir/data/train || exit 1;
  local/prepare_data.sh ${test_set} ${trn_trans} $out_dir/data/local/dict $out_dir/data/local/test $out_dir/data/test || exit 1;
fi

# L
if [ $stage -le 3 ]; then
  echo 'Start generate L'
  utils/prepare_lang.sh --position-dependent-phones false \
    $out_dir/data/local/dict "<UNK>" $out_dir/data/local/lang $out_dir/data/lang || exit 1;
  echo 'Finish generate L'
fi

# arpa LM
if [ $stage -le 4 ]; then
  echo 'Start generate LM'
  local/train_lms.sh \
      $out_dir/data/local/dict/lexicon.txt \
      $out_dir/data/local/train/text \
      $out_dir/data/local/lm || exit 1;
  echo 'Finish generate LM'
fi

# G compilation, check LG composition
if [ $stage -le 5 ]; then
  echo 'Start generate G'
  utils/format_lm.sh \
	  $out_dir/data/lang \
	  $out_dir/data/local/lm/3gram-mincount/lm_unpruned.gz \
	  $out_dir/data/local/dict/lexicon.txt \
	  $out_dir/data/lang_test || exit 1;
  echo 'Finish generate G'
fi

echo "local/prepare_all.sh succeeded"
exit 0;

