#!/bin/bash

stage=-1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 7 ]; then
  echo "$0 params error"
  exit 1;
fi

base_train=$1
base_test=$2
base_script=$3

extra_train=$4
extra_test=$5
extra_script=$6

out_dir=$7

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 1 ]; then
  local/prepare_dict.sh ${out_dir}/data/local/dict || exit 1;
fi

if [ $stage -le 2 ]; then
	local/prepare_data_base.sh \
		${out_dir}/data/local/dict/word_seg_vocab.txt \
		${base_train} \
		${base_test} \
		${base_script} \
		${out_dir}/data/local/base || exit 1;
        local/prepare_data_extra.sh \
                ${out_dir}/data/local/dict/word_seg_vocab.txt \
                ${extra_train} \
                ${extra_test} \
                ${extra_script} \
                ${out_dir}/data/local/extra || exit 1;
	for f in spk2utt utt2spk wav.scp text; do
		for data_set in train test; do
			tmp_out=${out_dir}/data/${data_set}
			mkdir -p $tmp_out
			cat ${out_dir}/data/local/base_${data_set}/$f ${out_dir}/data/local/extra_${data_set}/$f | sort -k 1 > ${tmp_out}/$f
		done
	done
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
      $out_dir/data/train/text \
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

