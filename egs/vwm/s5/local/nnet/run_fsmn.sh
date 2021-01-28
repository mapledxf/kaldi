out_dir=""
train=""
dev=""
stage=0
gmm_dir=""

. ./path.sh
. ./cmd.sh

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail
#########################
dnn_model=$1

data_fbk=$out_dir/data_fbank
##Make fbank features
if [ $stage -le 1 ]; then
    mkdir -p $data_fbk
    for x in $train $dev; do
        fbankdir=$out_dir/fbank/$x
        cp -r $out_dir/data/$x $data_fbk/$x
        steps/make_fbank.sh --nj 30 --cmd "$train_cmd"  --fbank-config conf/fbank.conf \
            $data_fbk/$x $out_dir/exp/make_fbank/$x $fbankdir
        steps/compute_cmvn_stats.sh $data_fbk/$x $out_dir/exp/make_fbank/$x $fbankdir
    done
fi
###############
if [ $stage -le 2 ]; then
    for x in $train $dev; do
    	steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
        	$out_dir/data/$x $out_dir/data/lang $gmm_dir $out_dir/exp/tri5a_ali_$x
    done
fi
#####CE-training

lrate=0.00001
dir=$out_dir/exp/tri7b_${dnn_model}
if [ $stage -le 3 ]; then
    proto=local/nnet/${dnn_model}.proto
    ori_num_pdf=`cat $proto |grep "Softmax" |awk '{print $3}'`
    echo $ori_num_pdf
    new_num_pdf=`gmm-info $gmm_dir/final.mdl |grep "number of pdfs" |awk '{print $4}'`
    echo $new_num_pdf
    new_proto=${proto}.$new_num_pdf
    sed -r "s/"$ori_num_pdf"/"$new_num_pdf"/g" $proto > $new_proto

    $cuda_cmd $dir/_train_nnet.log \
        local/nnet/train_faster.sh --learn-rate $lrate --nnet-proto $new_proto \
        --start_half_lr 5 --momentum 0.9 \
        --train-tool "nnet-train-fsmn-streams" \
        --feat-type plain --splice 1 \
        --cmvn-opts "--norm-means=true --norm-vars=false" --delta_opts "--delta-order=2" \
        --train-tool-opts "--minibatch-size=4096" \
        $data_fbk/$train \
        $data_fbk/$dev \
        $out_dir/data/lang \
        $out_dir/exp/tri5a_ali_$train \
        $out_dir/exp/tri5a_ali_$dev \
        $dir
fi
# ####Decode
acwt=0.08
# if [ $stage -le 4  ]; then
#         dataset="test_clean dev_clean test_other dev_other"
#         for set in $dataset
#         do
#              steps/nnet/decode.sh --nj 16 --cmd "$decode_cmd" \
#                  --acwt $acwt \
#                  $gmm_dir/graph_tgsmall \
#                  $data_fbk/$set $dir/decode_tgsmall_${set}

#              steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
#                  $data_fbk/$set $dir/decode_{tgsmall,tgmed}_${set}

#              steps/lmrescore_const_arpa.sh \
#                  --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
#                  $data_fbk/$set $dir/decode_{tgsmall,tglarge}_${set}

#              steps/lmrescore_const_arpa.sh \
#                  --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
#                  $data_fbk/$set $dir/decode_{tgsmall,fglarge}_${set}
#         done
#         for x in $dir/decode_*;
#         do
#                 grep WER $x/wer_* | utils/best_wer.sh
#         done
# fi
#gen ali & lat for smbr
nj=32
if [ $stage -le 5 ]; then
        steps/nnet/align.sh --nj $nj --cmd "$train_cmd" \
            $data_fbk/$train $out_dir/data/lang $dir ${dir}_ali
        steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --acwt $acwt \
            $data_fbk/$train $out_dir/data/lang $dir ${dir}_denlats
fi

####do smbr
if [ $stage -le 5 ]; then
        steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 2 --learn-rate 0.0000002 --acwt $acwt --do-smbr true \
            $data_fbk/$train $out_dir/data/lang $dir ${dir}_ali ${dir}_denlats ${dir}_smbr
fi
# ###decode
 dir=${dir}_smbr
 acwt=0.03
if [ $stage -le 6 ]; then
        for set in $dev 
        do
                steps/nnet/decode.sh --nj 16 --cmd "$decode_cmd" \
                --acwt $acwt \
                $gmm_dir/graph_tgsmall \
                $data_fbk/$set $dir/decode_tgsmall_${set}

                steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                $data_fbk/$set $dir/decode_{tgsmall,tgmed}_${set}

                steps/lmrescore_const_arpa.sh \
                --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
                $data_fbk/$set $dir/decode_{tgsmall,tglarge}_${set}

                steps/lmrescore_const_arpa.sh \
                --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
                $data_fbk/$set $dir/decode_{tgsmall,fglarge}_${set}
        done
        for x in $dir/decode_*;
        do
                grep WER $x/wer_* | utils/best_wer.sh
        done
fi

