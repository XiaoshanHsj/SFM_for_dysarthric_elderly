from datasets import load_dataset, load_metric
from datasets import ClassLabel, Audio
import random, os, sys
import pandas as pd
import numpy as np
import re, time, datetime
import json, librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2ConformerForCTC
from transformers import Wav2Vec2CTCTokenizer, AutoTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoModelForCTC
import torch
import torch.utils.data as data
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
import kaldiio
from kaldiio import WriteHelper

class BN_Dataset(data.Dataset):
    def __init__(self, path_list, cache_path, processor_path, model_path):
        self.feature = load_dataset('csv', data_files={'train':path_list}, cache_dir=cache_path)
        self.feature = self.feature.map(self.remove_special_characters)

        self.feature = self.feature.cast_column("file", Audio(sampling_rate=16000))
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).cuda()

        self.feature = self.feature.map(self.prepare_dataset, remove_columns=self.feature.column_names["train"], num_proc=1)

    def __getitem__(self, index):
        data = self.feature["train"][index]
        with torch.no_grad():
            input_values = torch.tensor(data["input_values"], device="cuda").unsqueeze(0)
            attention_mask = torch.tensor(data["attention_mask"], device="cuda").unsqueeze(0)
            output = self.model(input_values, attention_mask=attention_mask).bn_features
            y = list(t.cpu() for t in output)
            train_data = torch.stack(y, dim=0)
            # train_data = torch.squeeze(output)

        label_dur = int(data["utt_id"].split("_")[-1]) - int(data["utt_id"].split("_")[-2]) + 1
        return train_data, label_dur

    def __len__(self):
        return len(self.feature["train"])

    def remove_special_characters(self, batch):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
        return batch

    def prepare_dataset(self, batch):
        batch["utt_id"] = batch["id"]
        audio = batch["file"]
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["attention_mask"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).attention_mask[0]
        batch["input_length"] = len(batch["input_values"])

        return batch

def thread_testloaddata(path_list, cache_path, processor_path, model_path, file_num, shuffle):
    dataset = BN_Dataset(path_list, cache_path, processor_path, model_path)
    loader = data.DataLoader(
        dataset = dataset,
        batch_size = file_num,
        shuffle = shuffle,
        num_workers = 0,
        collate_fn = collate_data,
    )
    return loader

def collate_data(batch):
    data_list = []
    total_length = 0
    for train_data, data_length in batch:
        data_list.append(train_data)
        total_length = total_length + data_length
    data_numpy = torch.cat(tuple(data_list), 2)
    train_data = torch.FloatTensor(data_numpy)
    return train_data, total_length

def sampling(x, phone_label_dim):
    x_frames = x.size(0)
    delta_frames = x_frames - phone_label_dim
    if delta_frames == 0:
        return x
    elif delta_frames < 0:
        add_frame_num = phone_label_dim - x_frames
        left_add_frame_num = add_frame_num // 2
        right_add_frame_num = add_frame_num - left_add_frame_num

        left_add_frame = torch.zeros((left_add_frame_num, x.size(1)))
        left_add_frame[:] = x[0]

        right_add_frame = torch.zeros((right_add_frame_num, x.size(1)))
        right_add_frame[:] = x[-1]

        upsampling_data = torch.cat((left_add_frame, x, right_add_frame), 0)
        assert upsampling_data.size(0) == phone_label_dim
        return upsampling_data
    else:
        del_frame_num = x_frames - phone_label_dim
        left_del_frame_num = del_frame_num // 2
        right_del_frame_num = del_frame_num - left_del_frame_num
        x = x[left_del_frame_num : -right_del_frame_num, :]
        assert x.size(0) == phone_label_dim
        return x

if __name__ == '__main__':
    cache_path = sys.argv[1]
    processor_path = sys.argv[2]
    model_path = sys.argv[3]
    total_csv_list = sys.argv[4]
    absolute_path = os.getcwd()
    out_ark = absolute_path + "/bn_features_256_output/bn_feature.ark"
    out_scp = absolute_path + "/bn_features_256_output/bn_feature.scp"
    total_loader = thread_testloaddata(total_csv_list, cache_path, processor_path, model_path, 1, False)
    total_path_list = []
    with open(total_csv_list) as f_r:
        for line in f_r:
            if line.find("id,file,text") != -1:
                continue
            total_path_list.append(line.split(",")[0])
    total_output_dir = "bn_features_256_output"
    if not os.path.exists(total_output_dir):
        os.system("mkdir -p %s" % total_output_dir)
    with WriteHelper("ark,scp:%s,%s" % (out_ark, out_scp)) as writer:
        for step, (batch_x, batch_length) in enumerate(total_loader):
            utt_id = total_path_list[step]
            utt_x = batch_x.squeeze(0).permute(1, 0)
            bn_features = sampling(utt_x, batch_length)
            bn_features = bn_features.numpy()
            start = int(utt_id.split("_")[-2])
            end = int(utt_id.split("_")[-1])
            dur = end - start + 1
            assert bn_features.shape[0] == dur and bn_features.shape[1] == 256
            writer(utt_id, bn_features)
