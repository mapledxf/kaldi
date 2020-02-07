#!/bin/bash

# transform raw AISHELL-2 data to kaldi format

. ./path.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <word_seg_vocab> <train-dir> <test-dir> <script file> <tmp-dir>"
  echo " $0 word_seg_vocab.txt train_dir test_dir data_script data/local/train"
  exit 1;
fi

word_seg_vocab=$1
extra_train=$2
extra_test=$3
extra_script=$4
out_dir=$5

echo "$0: Preparing data in $out_dir"

python -c "import jieba" 2>/dev/null || \
        (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)

mkdir -p ${out_dir}_train
mkdir -p ${out_dir}_test

find $extra_train -iname "*.wav" > ${out_dir}_train/wav.flist
find $extra_test -iname "*.wav" > ${out_dir}_test/wav.flist

for data_set in train test; do
	tmp_dir=${out_dir}_${data_set}
	echo Preparing $tmp_dir transcriptions
	sed -e 's/\.wav//' ${tmp_dir}/wav.flist | awk -F '/' '{print $NF}' > ${tmp_dir}/utt.list_all
	awk '{printf("%s %s\n",$NF,substr($0,1,4))}' $tmp_dir/utt.list_all > $tmp_dir/utt2spk_all
	paste -d' ' $tmp_dir/utt.list_all $tmp_dir/wav.flist > $tmp_dir/wav.scp_all
        sed -e 's/\.wav//' $extra_script > $tmp_dir/transcripts.txt_all
        utils/filter_scp.pl -f 1 $tmp_dir/utt.list_all $tmp_dir/transcripts.txt_all | sort -k 1 | uniq > $tmp_dir/transcripts.txt
        awk '{print $1}' $tmp_dir/transcripts.txt > $tmp_dir/utt.list

	# wav.scp
	utils/filter_scp.pl -f 1 $tmp_dir/utt.list $tmp_dir/wav.scp_all | sort -k 1 | uniq > $tmp_dir/wav.scp
	# utt2spk & spk2utt
	utils/filter_scp.pl -f 1 $tmp_dir/utt.list $tmp_dir/utt2spk_all | sort -k 1 | uniq > $tmp_dir/utt2spk
	utils/utt2spk_to_spk2utt.pl $tmp_dir/utt2spk | sort -k 1 | uniq > $tmp_dir/spk2utt
	# text
	python2 local/word_segmentation.py ${word_seg_vocab} $tmp_dir/transcripts.txt > $tmp_dir/text
done
echo "$0 succeeded"
exit 0;
