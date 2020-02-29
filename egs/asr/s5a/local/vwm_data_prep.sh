#!/bin/bash
. ./path.sh || exit 1;
if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /noisy data/noisy"
  exit 1;
fi

vwm_audio_dir=$1/wav
vwm_text=$1/script/trans.txt

data=$2
train_dir=$data/local/train

mkdir -p $train_dir
mkdir -p $data/train

# data directory check
if [ ! -d $aidatatang_audio_dir ] || [ ! -f $aidatatang_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

echo "**** Creating VWM data folder ****"

find $vwm_audio_dir -iname "*.wav" > $train_dir/wav.flist
sed -e 's/\.wav//' $train_dir/wav.flist | awk -F '/' '{print $NF}' > $train_dir/utt.list_all
awk '{printf("%s %s\n",$NF,substr($0,1,4))}' $train_dir/utt.list_all > $train_dir/utt2spk_all
paste -d' ' $train_dir/utt.list_all $train_dir/wav.flist > $train_dir/wav.scp_all
sed -e 's/\.wav//' $vwm_text > $train_dir/transcripts.txt_all
utils/filter_scp.pl -f 1 $train_dir/utt.list_all $train_dir/transcripts.txt_all | sort -k 1 | uniq > $train_dir/transcripts.txt
awk '{print $1}' $train_dir/transcripts.txt > $train_dir/utt.list

#wav.scp
utils/filter_scp.pl -f 1 $train_dir/utt.list $train_dir/wav.scp_all | sort -k 1 | uniq > $train_dir/wav.scp
#utt2spk
utils/filter_scp.pl -f 1 $train_dir/utt.list $train_dir/utt2spk_all | sort -k 1 | uniq > $train_dir/utt2spk
#spk2utt
utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk | sort -k 1 | uniq > $train_dir/spk2utt
#text
python2 local/jieba_segment.py $train_dir/transcripts.txt > $train_dir/text
#cat $vwm_text |\
#  local/word_segment.py |\
#  awk '{if (NF > 1) print $0;}' > $train_dir/text

for f in spk2utt utt2spk wav.scp text; do
	cp $train_dir/$f $data/train/$f || exit 1;
done

utils/data/validate_data_dir.sh --no-feats $data/train || exit 1;

echo "$0: VWM noisy 48H data preparation succeeded"
exit 0;
