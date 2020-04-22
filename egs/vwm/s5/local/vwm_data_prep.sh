#!/bin/bash
. ./path.sh || exit 1;

src_dir=$1
out_dir=$2

vwm_audio_dir=$out_dir/wav
vwm_text=$out_dir/script/script.txt

data=$3
train_dir=$data/local/train

mkdir -p $train_dir
mkdir -p $data/train

echo "**** Creating VWM data folder ****"
echo "Formatting data"
[ -d $vwm_audio_dir ] &&  echo "$vwm_audio_dir exists, skip" || python local/vwm_format.py $src_dir $out_dir

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

echo "$0: VWM $out_dir data preparation succeeded"
exit 0;
