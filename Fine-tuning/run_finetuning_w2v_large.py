from datasets import load_dataset, load_metric
from datasets import ClassLabel, Audio
import random, jiwer
import pandas as pd
import numpy as np
import re, time
import json, librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, HubertForCTC, Data2VecAudioForCTC, WavLMForCTC, Wav2Vec2ConformerForCTC
from transformers import Wav2Vec2CTCTokenizer, AutoTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoModelForCTC
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper() + " "
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(batch):
    audio = batch["file"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["attention_mask"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).attention_mask[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def map_to_result(batch):
    model.eval()
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"], device="cuda").unsqueeze(0)
        logits = model(input_values, attention_mask=attention_mask).logits

    model.train()
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    # label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    results = dbank["evde"].map(map_to_result, remove_columns=dbank["evde"].column_names)
    measures = jiwer.compute_measures(results["text"], results["pred_str"])
    return {"wer": measures["wer"]}

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

train_csv = "data/train/train.csv"
evde_csv = "data/test/test.csv"
dbank = load_dataset('csv', data_files={'train':train_csv, 'evde':evde_csv}, cache_dir="cache")
dbank = dbank.map(remove_special_characters)

processor_path = "facebook/wav2vec2-large-960h-lv60"
processor = Wav2Vec2Processor.from_pretrained(processor_path)
dbank = dbank.cast_column("file", Audio(sampling_rate=16000))

dbank = dbank.map(prepare_dataset, remove_columns=dbank.column_names["train"], num_proc=4)

# min_input_length_in_sec = 1.0
# dbank["train"] = dbank["train"].filter(lambda x: x > min_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")
model_path = "facebook/wav2vec2-large-960h-lv60"
model = Wav2Vec2ForCTC.from_pretrained(
    model_path,
    ctc_loss_reduction="mean",
    vocab_size=len(processor.tokenizer.get_vocab()),
    pad_token_id=processor.tokenizer.pad_token_id,
    cache_dir="large_model_cache",
)

model.config.ctc_zero_infinity = True
model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir="dbank_output_wav2vec2_3e-5",
    group_by_length=True,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True,
    # eval_delay=90000,
    save_steps=3000,
    eval_steps=3000,
    logging_steps=3000,
    learning_rate=3e-5,
    weight_decay=0.005,
    warmup_steps=1000,
    dataloader_num_workers=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dbank["train"],
    eval_dataset=dbank["evde"],
    tokenizer=processor.feature_extractor,
)

trainer.train()

