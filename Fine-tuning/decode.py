import contextlib
from datasets import load_dataset, load_metric
from datasets import ClassLabel, Audio
import random, sys
import pandas as pd
import numpy as np
import re, time, math
import json, librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, HubertForCTC, Data2VecAudioForCTC, WavLMForCTC
from transformers import Wav2Vec2CTCTokenizer, AutoTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoModelForCTC
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM
import jiwer

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper() + " "
    return batch

def prepare_dataset(batch):
    audio = batch["file"]
    batch["utt_id"] = batch["id"]
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["attention_mask"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).attention_mask[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"], device="cuda").unsqueeze(0)
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["uttid"] = batch["utt_id"]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    return batch

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

evde_csv = sys.argv[1]
dbank = load_dataset('csv', data_files={'evde':evde_csv}, cache_dir="cache")
dbank = dbank.map(remove_special_characters)

dbank = dbank.cast_column("file", Audio(sampling_rate=16000))

processor_path = "facebook/wav2vec2-large-960h-lv60"
model_path = sys.argv[2]
processor = Wav2Vec2Processor.from_pretrained(processor_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path).cuda()

dbank = dbank.map(prepare_dataset, remove_columns=dbank["evde"].column_names, num_proc=4)
wer_metric = load_metric("wer")

results = dbank["evde"].map(map_to_result, remove_columns=dbank["evde"].column_names)
print("text\tpred_str")
for i in range(len(results["text"])):
    print(results["text"][i]+"\t"+results["pred_str"][i])
print("Test WER: {:.9f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
measures = jiwer.compute_measures(results["text"], results["pred_str"])
print("wer:", measures['wer'])
print("substitutions:", measures['substitutions'])
print("deletions:", measures['deletions'])
print("insertions:", measures['insertions'])
