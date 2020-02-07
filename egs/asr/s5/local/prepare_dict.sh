#!/bin/bash

# This is a shell script, and it download and process DaCiDian for Mandarin ASR.

. ./path.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <dict-dir>";
  exit 1;
fi

dir=$1
download_dir=$dir/BigCiDian
# download the DaCiDian from github
if [ ! -d $download_dir ]; then
  git clone https://github.com/mapledxf/BigCiDian.git $download_dir
fi

# here we map <UNK> to the phone spn(spoken noise)
mkdir -p $dir
python $download_dir/utils/convert_pinyin_chart_to_mapping.py \
	$download_dir/CN/pinyin_chart.csv \
	$dir/pinyin_to_phone.txt
python $download_dir/utils/DaCiDian.py \
	$download_dir/CN/word_to_pinyin.txt \
	$dir/pinyin_to_phone.txt > $dir/CN.txt

iconv -f ISO_8859-10 -t utf8 ${download_dir}/EN/cmudict-0.7b.txt >$dir/tmp || exit 1;
python $download_dir/utils/map_arpa_to_ipa.py \
	$download_dir/EN/ARPA2IPA.map \
	$dir/tmp \
	$dir/EN.txt || exit 1;
cat $dir/EN.txt $dir/CN.txt | sort -u > $dir/lexicon.txt
echo -e "<UNK>\tspn" >> $dir/lexicon.txt

python $download_dir/utils/dict_to_phoneset.py $dir/lexicon.txt $dir/nonsilence_phones.txt
# prepare silence_phones.txt, nonsilence_phones.txt, optional_silence.txt, extra_questions.txt

echo sil > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

cat $dir/silence_phones.txt | awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; if($p eq "\$0"){$q{""} .= "$p ";}else{$q{$2} .= "$p ";} } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

# jieba's vocab format requires word count(frequency), set to 99
awk '{print $1}' $dir/lexicon.txt | sort | uniq | awk '{print $1,88}'> $dir/word_seg_vocab.txt
echo "local/prepare_dict.sh succeeded"
exit 0;
