#!/bin/bash

# transform raw AISHELL-2 data to kaldi format

. ./path.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <wav-dir> <script file> <dict-dir> <tmp-dir> <output-dir>"
  echo " $0 wav_dir data_script data/local/dict data/local/train data/train"
  exit 1;
fi

wav_dir=$1
script=$2
dict_dir=$3
tmp=$4
dir=$5

echo "prepare_data.sh: Preparing data in $script"

mkdir -p $tmp
mkdir -p $dir

sed -e 's/\.wav//' $script > $dict_dir/trans.txt
find $wav_dir -iname "*.wav" > $dict_dir/wav.list
sed -e 's/\.wav//' $dict_dir/wav.list | awk -F '/' '{print $NF}' > $dict_dir/utt.list
paste -d' ' $dict_dir/utt.list $dict_dir/wav.list > $dict_dir/wav.scp

# validate utt-key list
awk '{print $1}' $dict_dir/wav.scp   > $tmp/wav_utt.list
awk '{print $1}' $dict_dir/trans.txt > $tmp/trans_utt.list
utils/filter_scp.pl -f 1 $tmp/wav_utt.list $tmp/trans_utt.list > $tmp/utt.list

# wav.scp
cp $dict_dir/wav.scp $tmp/tmp_wav.scp 
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_wav.scp | sort -k 1 | uniq > $tmp/wav.scp

# text
python -c "import jieba" 2>/dev/null || \
  (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)
utils/filter_scp.pl -f 1 $tmp/utt.list $dict_dir/trans.txt | sort -k 1 | uniq > $tmp/trans.txt
# jieba's vocab format requires word count(frequency), set to 99
awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $tmp/word_seg_vocab.txt
python2 local/word_segmentation.py $tmp/word_seg_vocab.txt $tmp/trans.txt > $tmp/text

# utt2spk & spk2utt
awk '{printf("%s %s\n",$NF,substr($0,1,4))}' $tmp/utt.list > $tmp/tmp_utt2spk
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_utt2spk | sort -k 1 | uniq > $tmp/utt2spk
utils/utt2spk_to_spk2utt.pl $tmp/utt2spk | sort -k 1 | uniq > $tmp/spk2utt

# copy prepared resources from tmp_dir to target dir
mkdir -p $dir
for f in wav.scp text spk2utt utt2spk; do
  cp $tmp/$f $dir/$f || exit 1;
done

echo "local/prepare_data.sh succeeded"
exit 0;
