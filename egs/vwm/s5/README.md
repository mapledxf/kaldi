# VWM ASR Model
## Datasets used
* Aidatatang (140 hours)
* Aishell (151 hours)
* MagicData (712 hours)
* Primewords (99 hours)
* ST-CMDS (110 hours)
* THCHS-30 (26 hours)


## Train Commands
```
CUDA_VISIBLE_DEVICES=1 nohup ./run.sh --stage 0 > train.log 2>&1 &
```

oputput dir: 
```
/data/xfding/train_result/asr/ali/exp/tri7b_DFSMN_S_denlats
/data/xfding/train_result/asr/ali/exp/tri7b_DFSMN_S_smbr
```

## Vosk
### File structure
There are two formats to use vosk:
1.
```
|____mfcc.conf
|____HCLG.fst
|____final.mdl
|____README
|____ivector
| |____global_cmvn.stats
| |____splice.conf
| |____final.mat
| |____online_cmvn.conf
| |____final.ie
| |____final.dubm
|____phones.txt
|____word_boundary.int
|____words.txt
```
2.
```
____mfcc.conf
|____.DS_Store
|____HCLr.fst
|____disambig_tid.int
|____final.mdl
|____README
|____ivector
| |____global_cmvn.stats
| |____splice.conf
| |____final.mat
| |____online_cmvn.conf
| |____final.ie
| |____final.dubm
|____word_boundary.int
|____Gr.fst
```

### Create files

./local/lookahead/run_lookahead.sh
mkdir am
mv final.mdl am/
mv graph_lookahead or graph_lookahead_arpa to graph
mv ivector_extractor/ to ivector
cp conf/splice.conf ivector
cp mfcc.conf
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
