# SFM_for_dysarthric_elderly

This repository is the open source code for our ICASSP 2023 paper and IEEE/ACM TASLP:

[Exploring Self-Supervised Pre-Trained ASR Models for Dysarthric and Elderly Speech Recognition](https://ieeexplore.ieee.org/abstract/document/10097275)

[Self-Supervised ASR Models and Features for Dysarthric and Elderly Speech Recognition](https://ieeexplore.ieee.org/abstract/document/10584335)

![model](figs/models.png)

## Code

### SFM fine-tuning
You can find an example script for fine-tuning the Wav2Vec2.0 model in `Fine-tuning/run_finetuning_w2v_large.py`.

### Inference
`Fine-tuning/decode.py`

### SSL Speech Representation Extraction 
1. Bottleneck Module  
`Fine-tuning/Bottleneck_module.py`  
This bottleneck module can be inserted either at the positions investigated in our paper or at alternative locations.  
2. Fine-tuning  
You need to add the dim hyperparameter in config `transformers/models/wav2vec2/configuration_wav2vec2.py`, then in the training script,
```
model = Wav2Vec2ForCTC.from_pretrained(
    model_path,
    ctc_loss_reduction="mean",
    vocab_size=len(processor.tokenizer.get_vocab()),
    pad_token_id=processor.tokenizer.pad_token_id,
    cache_dir="large_model_cache",
    dim=256,  # the dim of bottleneck features
)
```
also ignore the bn_features during inference, as 
```
model.config.keys_to_ignore_at_inference = ["bn_features"]
```

3. Extraction
After fine-tuning the SFM with Bottleneck Module, you can refer `Fine-tuning/get_bn_feature.py` script to extract SSL speech features.  

### A2A Inversion
You can refer to the code in the `A2A_inversion` directory, where you will need to prepare `*.scp` files.  
To train the A2A inversion model, run:
```
python3 A2A_inversion/main.py --train
```
To generate articulatory features, run:
```
python3 A2A_inversion/main.py --test
```

### SFM integration
For Kaldi recipes, please refer to [this recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/swbd/s5c/local/chain/tuning/run_tdnn_7q.sh).  
For Conformer recipes, please refer to [this recipe](https://github.com/espnet/espnet/blob/master/egs/swbd/asr1/run.sh) and [this yaml](https://github.com/espnet/espnet/blob/master/egs/swbd/asr1/conf/tuning/train_pytorch_conformer.yaml).  

**Time-synchronous frame-level joint decoding**  
For implementation details, please refer to [this script](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet3/decode_score_fusion.sh).  
This script can be easily extended to support fusion of more than two hybrid TDNN systems.

**Cross-system multi-pass decoding**
1. Get the N-best hypotheses  
Here are examples for Kaldi nnet3 TDNN and Conformer  
Kaldi:  
```
cat $dir/lat.*.gz |\
gunzip -c - |\
    lattice-to-nbest --acoustic-scale=0.1 --n=30 ark:- ark:$dir/nbest/30best

nbest-to-linear ark,t:$dir/nbest/30best ark,t:$dir/nbest/ali ark,t:$dir/nbest/words \
        ark,t:$dir/nbest/lmscore ark,t:$dir/nbest/acscore
cat $dir/nbest/words | utils/int2sym.pl -f 2- \
        $lang/words.txt > $dir/nbest/words.txt
```
Conformer:  
```
${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --nbest 30 \
```
2. Obtain SFM Scores  
create `*.csv` files based on the N-best hypotheses, then input these hypotheses into the fine-tuned SFM to compute the CTC loss, which will be used as the SFM scores.
3. Obtain the final interpolated scores.   
You can refer to the script `SFM_integration/get_w2v_tdnn_score_and_sort.py` to interpolate the scores of different systems.

## Ciataions

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and citations.

```bibtex
@INPROCEEDINGS{10097275,
  author={Hu, Shujie and Xie, Xurong and Jin, Zengrui and Geng, Mengzhe and Wang, Yi and Cui, Mingyu and Deng, Jiajun and Liu, Xunying and Meng, Helen},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Exploring Self-Supervised Pre-Trained ASR Models for Dysarthric and Elderly Speech Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10097275}}

@ARTICLE{10584335,
  author={Hu, Shujie and Xie, Xurong and Geng, Mengzhe and Jin, Zengrui and Deng, Jiajun and Li, Guinan and Wang, Yi and Cui, Mingyu and Wang, Tianzi and Meng, Helen and Liu, Xunying},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Self-Supervised ASR Models and Features for Dysarthric and Elderly Speech Recognition}, 
  year={2024},
  volume={32},
  number={},
  pages={3561-3575},
  doi={10.1109/TASLP.2024.3422839}}
