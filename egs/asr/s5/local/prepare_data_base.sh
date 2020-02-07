#!/bin/bash

# Copyright 2017 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <word_seg_vocab> <train-dir> <test-dir> <script file> <tmp-dir>"
  echo " $0 word_seg_vocab.txt train_dir test_dir data_script data/local/train"
  exit 1;
fi

word_seg_vocab=$1
base_train=$2
base_test=$3
base_script=$4
out_dir=$5

echo "$0: Preparing data in $out_dir"

python -c "import jieba" 2>/dev/null || \
	(echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)

mkdir -p ${out_dir}_train
mkdir -p ${out_dir}_test

# find wav audio file for train, dev and test resp.
find $base_train -iname "*.wav" > ${out_dir}_train/wav.flist
find $base_test -iname "*.wav" > ${out_dir}_test/wav.flist

# Transcriptions preparation
for train_set in train test; do
	tmp_dir=${out_dir}_${train_set}
	echo Preparing $tmp_dir transcriptions
	sed -e 's/\.wav//' $tmp_dir/wav.flist | awk -F '/' '{print $NF}' > $tmp_dir/utt.list_all
	sed -e 's/\.wav//' $tmp_dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $tmp_dir/utt2spk_all
	paste -d' ' $tmp_dir/utt.list_all $tmp_dir/wav.flist > $tmp_dir/wav.scp_all
	utils/filter_scp.pl -f 1 $tmp_dir/utt.list_all $base_script | sort -k 1 | uniq > $tmp_dir/transcripts.txt
	awk '{print $1}' $tmp_dir/transcripts.txt > $tmp_dir/utt.list

	utils/filter_scp.pl -f 1 $tmp_dir/utt.list $tmp_dir/wav.scp_all | sort -k 1 | uniq > $tmp_dir/wav.scp
	utils/filter_scp.pl -f 1 $tmp_dir/utt.list $tmp_dir/utt2spk_all | sort -k 1 | uniq > $tmp_dir/utt2spk
	utils/utt2spk_to_spk2utt.pl $tmp_dir/utt2spk > $tmp_dir/spk2utt

	# text
        python2 local/word_segmentation.py ${word_seg_vocab} $tmp_dir/transcripts.txt > $tmp_dir/text
done

echo "$0: base data preparation succeeded"
exit 0;
