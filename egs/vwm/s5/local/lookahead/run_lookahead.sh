#!/bin/bash

. ./path.sh

# Example script for lookahead composition
out_dir=/data/xfding/train_result/asr/vwm
lm=${out_dir}/data/local/lm/3gram-mincount/lm_unpruned.gz
am=${out_dir}/exp/chain_cleaned/tdnn_cnn_1a_sp_online

if [ ! -f "${KALDI_ROOT}/tools/openfst/lib/libfstlookahead.so" ]; then
    echo "Missing ${KALDI_ROOT}/tools/openfst/lib/libfstlookahead.so"
    echo "Make sure you compiled openfst with lookahead support. Run make in ${KALDI_ROOT}/tools after git pull."
    exit 1
fi
if [ ! -f "${KALDI_ROOT}/tools/openfst/bin/ngramread" ]; then
    echo "You appear to not have OpenGRM tools installed. Missing ${KALDI_ROOT}/tools/openfst/bin/ngramread"
    echo "cd to $KALDI_ROOT/tools and run extras/install_opengrm.sh."
    exit 1
fi
export LD_LIBRARY_PATH=${KALDI_ROOT}/tools/openfst/lib/fst

# Baseline
echo "Start baseline"
utils/format_lm.sh ${out_dir}/data/lang_chain ${lm} \
    ${out_dir}/data/local/dict/lexicon.txt ${out_dir}/data/lang_test_arpa_base

echo "Start lookahead"
utils/mkgraph_lookahead.sh --self-loop-scale 1.0 --remove-oov --compose-graph \
    ${out_dir}/data/lang_test_arpa_base ${am} ${am}/graph_lookahead

echo "Start arpa lookahead"
# Compile arpa graph
utils/mkgraph_lookahead.sh --self-loop-scale 1.0 --compose-graph \
    ${out_dir}/data/lang_test_arpa_base ${am} ${lm} ${am}/graph_lookahead_arpa
