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
For Kaldi recipes, please refer to [this scripts](https://github.com/kaldi-asr/kaldi/blob/master/egs/swbd/s5c/local/chain/tuning/run_tdnn_7q.sh).  
For Conformer recipes, please refer to [this yaml](https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/train_asr_conformer.yaml).
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
