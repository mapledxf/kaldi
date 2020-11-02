This is a Chinese speech recognition recipe that trains on all Chinese corpora on [OpenSLR](http://www.openslr.org), including:
* Aidatatang (140 hours)
* Aishell (151 hours)
* MagicData (712 hours)
* Primewords (99 hours)
* ST-CMDS (110 hours)
* THCHS-30 (26 hours)

This recipe was developed by Xingyu Na (Microsoft Corporation) and Hui Bu (AISHELL Foundation).

## Highlights

1. This recipe start from bootstraping small GMM models using small portion of data to speaker adaptive training using cleaned full partition, which is over 1k hours.
2. A general lexicon is prepared by combining CMU English dictionary and CC-CEDIT Chinese dictionary, then **expanded using G2P**.
3. A general language model is trained using all training transcriptions, while **corpus specific LMs** are optionally obtained by interpolated with the general LM.
4. Features are extracted in an online fashion.
5. A Chain model ready for **online ASR** is trained, prepared and evaluated.
6. Data preparation scripts are copied from existing recipes, so it is straightforward for any user to **expand the corpora**.

CUDA_VISIBLE_DEVICES=1 nohup ./run.sh --stage 0 > train.log 2>&1 &
oputput dir: /data/xfding/train_result/asr/vwm/exp/chain_cleaned/tdnn_cnn_1a_sp_online

vosk: 
./local/lookahead/run_lookahead.sh
mkdir am
mv final.mdl am/
mv graph_lookahead or graph_lookahead_arpa to graph
mv ivector_extractor/ to ivector
cp conf/splice.conf ivector
create conf/model.conf:
--min-active=200
--max-active=3000
--beam=10.0
--lattice-beam=2.0
--acoustic-scale=1.0
--frame-subsampling-factor=3
--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10
--endpoint.rule2.min-trailing-silence=0.5
--endpoint.rule3.min-trailing-silence=1.0
--endpoint.rule4.min-trailing-silence=2.0
