#!/bin/bash

set -e

# number of jobs
nj=20
stage=0

out_dir=$1

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh;
. ./utils/parse_options.sh

# Now make MFCC features.
if [ $stage -le 1 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  echo "$0: stage $stage train mfcc"
  for x in train test; do
    steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj $nj \
	    $out_dir/data/$x \
	    $out_dir/exp/make_mfcc/$x \
	    $out_dir/mfcc || exit 1;
    steps/compute_cmvn_stats.sh \
	    $out_dir/data/$x \
	    $out_dir/exp/make_mfcc/$x \
	    $out_dir/mfcc || exit 1;
    utils/fix_data_dir.sh $out_dir/data/$x || exit 1;
  done
fi

# mono
if [ $stage -le 2 ]; then
  echo "$0: stage $stage train mono"
  # training
  steps/train_mono.sh --cmd "$train_cmd" --nj $nj \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/mono || exit 1;
  echo "$0: stage $stage align  mfcc"

  # decoding
  utils/mkgraph.sh \
    $out_dir/data/lang_test \
    $out_dir/exp/mono \
    $out_dir/exp/mono/graph || exit 1;
#  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
#    $out_dir/exp/mono/graph \
#    $out_dir/data/test \
#    $out_dir/exp/mono/decode_test
  
  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/mono \
    $out_dir/exp/mono_ali || exit 1;
fi 

# tri1
if [ $stage -le 3 ]; then
  echo "$0: stage $stage train tri1"
  # training
  steps/train_deltas.sh --cmd "$train_cmd" \
   2500 20000 \
   $out_dir/data/train \
   $out_dir/data/lang \
   $out_dir/exp/mono_ali \
   $out_dir/exp/tri1 || exit 1;
  echo "$0: stage $stage align tri1"

  # decoding
  utils/mkgraph.sh \
    $out_dir/data/lang_test \
    $out_dir/exp/tri1 \
    $out_dir/exp/tri1/graph || exit 1;
#  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
#    $out_dir/exp/tri1/graph \
#    $out_dir/data/test \
#    $out_dir/exp/tri1/decode_test

  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri1 \
    $out_dir/exp/tri1_ali || exit 1;
fi

# tri2
if [ $stage -le 4 ]; then
  echo "$0: stage $stage train tri2"
  # training
  steps/train_deltas.sh --cmd "$train_cmd" \
   2500 20000 \
   $out_dir/data/train \
   $out_dir/data/lang \
   $out_dir/exp/tri1_ali \
   $out_dir/exp/tri2 || exit 1;
  echo "$0: stage $stage align tri2"

  # decoding
  utils/mkgraph.sh \
    $out_dir/data/lang_test \
    $out_dir/exp/tri2 \
    $out_dir/exp/tri2/graph || exit 1;
#  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
#    $out_dir/exp/tri2/graph \
#    $out_dir/data/test \
#    $out_dir/exp/tri2/decode_test

  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri2 \
    $out_dir/exp/tri2_ali || exit 1;
fi

# tri3
if [ $stage -le 5 ]; then
  echo "$0: stage $stage train tri3 (lda+mllt)"
  # training [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    2500 20000 \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri2_ali \
    $out_dir/exp/tri3 || exit 1;

  # decoding
  utils/mkgraph.sh \
    $out_dir/data/lang_test \
    $out_dir/exp/tri3 \
    $out_dir/exp/tri3/graph || exit 1;
#  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
#    $out_dir/exp/tri3/graph \
#    $out_dir/data/test \
#    $out_dir/exp/tri3/decode_test

  # alignment with fMLLR
  echo "$0: stage $stage align tri3 with fmllr"
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri3 \
    $out_dir/exp/tri3_ali || exit 1;
fi

# tri4
if [ $stage -le 6 ]; then
  echo "$0: stage $stage train tri4"
  # train with sat
  steps/train_sat.sh --cmd "$train_cmd" \
    2500 20000 \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri3_ali \
    $out_dir/exp/tri4 || exit 1;

  # decoding
  utils/mkgraph.sh \
    $out_dir/data/lang_test \
    $out_dir/exp/tri4 \
    $out_dir/exp/tri4/graph || exit 1;
#  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $nj \
#    $out_dir/exp/tri4/graph \
#    $out_dir/data/test \
#    $out_dir/exp/tri4/decode_test

  # aligment
  echo "$0: stage $stage align tri4 with fmllr"
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri4 \
    $out_dir/exp/tri4_ali || exit 1;
fi

# tri5
if [ $stage -le 7 ]; then
  echo "$0: stage $stage train tri5 (sat)"
  # Building a larger SAT system.
  steps/train_sat.sh --cmd "$train_cmd" \
    3500 100000 \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri4_ali \
    $out_dir/exp/tri5 || exit 1;

  # decoding
  utils/mkgraph.sh \
    $out_dir/data/lang_test \
    $out_dir/exp/tri5 \
    $out_dir/exp/tri5/graph || exit 1;
#  steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $nj \
#    $out_dir/exp/tri5/graph \
#    $out_dir/data/test \
#    $out_dir/exp/tri5/decode_test

  # aligment
  echo "$0: stage $stage align tri5 with fmllr"
  steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
    $out_dir/data/train \
    $out_dir/data/lang \
    $out_dir/exp/tri5 \
    $out_dir/exp/tri5_ali || exit 1;
fi

echo "local/run_gmm.sh succeeded"
exit 0;

