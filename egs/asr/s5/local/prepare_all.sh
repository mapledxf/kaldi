#!/bin/bash

stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "prepare_all.sh <corpus-train-dir> <corpus-train-transcript> <output-dir>"
  echo " e.g prepare_all.sh /data/train/wav /data/train/trans.txt /data/output"
  exit 1;
fi

trn_set=$1
trn_trans=$2
out_dir=$3

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 1 ]; then
  local/prepare_dict.sh ${out_dir}/data/local/dict || exit 1;
fi

sed -e 's/\.wav//' $trn_trans > $out_dir/data/local/dict/trans.txt
find $trn_set -iname "*.wav" > $out_dir/data/local/dict/wav.list
sed -e 's/\.wav//' $out_dir/data/local/dict/wav.list | awk -F '/' '{print $NF}' > $out_dir/data/local/dict/utt.list
paste -d' ' $out_dir/data/local/dict/utt.list $out_dir/data/local/dict/wav.list > $out_dir/data/local/dict/wav.scp

local/prepare_data.sh ${trn_trans} $out_dir/data/local/dict $out_dir/data/local/train $out_dir/data/train || exit 1;

# L
if [ $stage -le 3 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    $out_dir/data/local/dict "<UNK>" $out_dir/data/local/lang $out_dir/data/lang || exit 1;
fi

# arpa LM
if [ $stage -le 4 ]; then
  local/train_lms.sh \
      $out_dir/data/local/dict/lexicon.txt \
      $out_dir/data/local/train/text \
      $out_dir/data/local/lm || exit 1;
fi

 G compilation, check LG composition
if [ $stage -le 5 ]; then
  utils/format_lm.sh \
	  $out_dir/data/lang \
	  $out_dir/data/local/lm/3gram-mincount/lm_unpruned.gz \
	  $out_dir/data/local/dict/lexicon.txt \
	  $out_dir/data/lang_test || exit 1;
fi

echo "local/prepare_all.sh succeeded"
exit 0;

