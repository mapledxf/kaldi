#!/bin/bash

# This script is based on run_tdnn_1a.sh.
# This setup used online pitch to train the neural network.
# It requires a online_pitch.conf in the conf dir.

set -e

# configs for 'chain'
affix=
stage=0
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
num_epochs=5
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=1
num_jobs_final=1
nj=20
minibatch_size=128
frames_per_eg=150,110,90
remove_egs=true
common_egs_dir=
xent_regularize=0.1

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

outdir=$1

dir=$outdir/exp/tdnn_vwm${affix:+_$affix}_sp
train_set=train_sp
ali_dir=$outdir/exp/tri5_sp_ali
treedir=$outdir/exp/tri6_7d_tree_sp
lang=$outdir/data/lang_chain

if [ $stage -le 6 ]; then
  echo 'Start generate ivector'
  local/nnet3/run_ivector_common.sh --stage $stage --online true $outdir || exit 1;
fi

# if we are using the speed-perturbed data we need to generate
# alignments for it.
if [ $stage -le 7 ]; then
  echo 'Start generate sp lats ali'
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    $outdir/data/$train_set \
    $outdir/data/lang \
    $outdir/exp/tri5 \
    $outdir/exp/tri5_sp_lats || exit 1;
  rm $outdir/exp/tri5_sp_lats/fsts.*.gz # save space
fi

if [ $stage -le 8 ]; then
  echo 'Start generate topo'
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r $outdir/data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist > $lang/topo
fi

if [ $stage -le 9 ]; then
  echo 'Start generate tree'
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" \
      5000 \
      $outdir/data/$train_set \
      $lang \
      $ali_dir \
      $treedir
fi

if [ $stage -le 10 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 11 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $outdir/exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --use-gpu wait \
    --feat-dir $outdir/data/${train_set}_hires_online \
    --tree-dir $treedir \
    --lat-dir $outdir/exp/tri5_sp_lats \
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph
if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 \
	  $outdir/data/lang_test \
	  $dir \
	  $graph_dir
fi

#if [ $stage -le 13 ]; then
#	steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
#		--nj 10 --cmd "$decode_cmd" \
#		--online-ivector-dir $outdir/exp/nnet3/ivectors_test \
#		$graph_dir \
#		$outdir/data/test_hires_online $dir/decode_test || exit 1;
#fi

if [ $stage -le 14 ]; then
	steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
		--add-pitch true \
		$lang \
		$outdir/exp/nnet3/extractor "$dir" ${dir}_online || exit 1;
	cp $graph_dir/HCLG.fst ${dir}_online/HCLG.fst
	cp $graph_dir/words.txt ${dir}_online/words.txt
fi

#dir=${dir}_online
#if [ $stage -le 15 ]; then
#	steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
#		--nj 10 --cmd "$decode_cmd" \
#		--config conf/decode.conf \
#		$graph_dir \
#		$outdir/data/test_hires_online $dir/decode_test || exit 1;
#fi
#
#if [ $stage -le 16 ]; then
#	steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
#		--nj 10 --cmd "$decode_cmd" --per-utt true \
#		--config conf/decode.conf \
#		$graph_dir \
#		$outdir/data/test_hires_online $dir/decode_test_per_utt || exit 1;
#fi
exit;
