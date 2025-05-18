import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from itertools import chain
from config import LABELS
from transformers import (
    XLMRobertaTokenizerFast, XLMRobertaForTokenClassification,
    BertTokenizerFast, BertConfig, BertForTokenClassification,
    AutoTokenizer, AutoConfig, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

"""
The training pipeline follows these steps for all three models:

Load and preprocess the input CSV:
   - Idiom spans are marked using token indices.
   - BIO tags ("B", "I", "O") are generated based on these indices.

Prepare HuggingFace datasets:
   - Sentences are tokenized using the appropriate tokenizer.
   - BIO labels are aligned to subword tokens using word_ids mapping.

Define model and training setup:
   - A pre-trained transformer is loaded and configured for 3-class token classification.
   - Class weights are computed to adjust the loss function.
   - Training arguments include learning rate, batch size, weight decay, and early stopping.

Train with HuggingFace's Trainer:
   - Evaluation is performed on the validation set each epoch.
   - The best model (lowest eval loss) is saved.

Save results:
   - Trained models and tokenizers are saved for future inference.

All models share this pipeline structure, with differences only in the language-specific tokenizer.
"""

# This function creates BIO tags from idiom index positions
def get_bio_tags(tokens, idiom_indices):
    bio = ["O"] * len(tokens)
    if idiom_indices != [-1]:
        for i, idx in enumerate(idiom_indices):
            bio[idx] = "B" if i == 0 else "I"
    return bio


# Fine-tunes xlm-roberta-large 
def train_xlmr(df):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
    label_to_id = {l: i for i, l in enumerate(LABELS)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    dataset = {"tokens": [], "labels": [], "language": [], "id": []}
    for _, row in df.iterrows():
        tokens = row["tokenized_sentence"]
        indices = row["indices"]
        dataset["tokens"].append(tokens)
        dataset["labels"].append(get_bio_tags(tokens, indices))
        dataset["language"].append(row["language"])
        dataset["id"].append(row["id"])
    df = pd.DataFrame(dataset)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["language"])

    hf_train = Dataset.from_pandas(train_df)
    hf_val = Dataset.from_pandas(val_df)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=True)
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = [label_to_id[label[word_id]] if word_id is not None else -100 for word_id in word_ids]
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    tokenized_train = hf_train.map(tokenize_and_align_labels, batched=True, remove_columns=hf_train.column_names)
    tokenized_val = hf_val.map(tokenize_and_align_labels, batched=True, remove_columns=hf_val.column_names)

    model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-large", num_labels=len(LABELS), id2label=id_to_label, label2id=label_to_id)

    args = TrainingArguments(
        output_dir="xlmr_large_idiom_finetuned",
        learning_rate=1e-5,
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        weight_decay=0.001,
        label_smoothing_factor=0.1,
        warmup_steps=200,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    model.save_pretrained("xlmr_large_idiom_finetuned")
    tokenizer.save_pretrained("xlmr_large_idiom_finetuned")


# Fine-tunes dbmdz/bert-base-turkish-cased
def train_berturk(df):
    label_to_id = {l: i for i, l in enumerate(LABELS)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    df = df[df["language"] == "tr"]
    train_df, val_df = train_test_split(df, test_size=0.1)

    def prepare_dataset(data):
        return Dataset.from_dict({
            "tokens": data["tokenized_sentence"].tolist(),
            "labels": [get_bio_tags(row["tokenized_sentence"], row["indices"]) for _, row in data.iterrows()]
        })

    tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-cased")
    train_ds = prepare_dataset(train_df)
    val_ds = prepare_dataset(val_df)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=True)
        tokenized["labels"] = [[label_to_id[label[word_id]] if word_id is not None else -100 for word_id in tokenized.word_ids(i)] for i, label in enumerate(examples["labels"])]
        return tokenized

    train_tok = train_ds.map(tokenize_and_align_labels, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_and_align_labels, batched=True, remove_columns=val_ds.column_names)

    flat_labels = list(chain.from_iterable(train_tok["labels"]))
    label_counts = Counter(l for l in flat_labels if l != -100)
    total = sum(label_counts.values())
    class_weights = torch.tensor([1.0 - (label_counts.get(i, 0) / total) for i in range(len(LABELS))])

    class CustomBert(BertForTokenClassification):
        def __init__(self, config, weights=None):
            super().__init__(config)
            self.class_weights = weights

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            if labels is not None and self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device), ignore_index=-100)
                loss = loss_fct(outputs.logits.view(-1, self.config.num_labels), labels.view(-1))
                return {"loss": loss, "logits": outputs.logits}
            return outputs

    config = BertConfig.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=len(LABELS), id2label=id_to_label, label2id=label_to_id)
    model = CustomBert.from_pretrained("dbmdz/bert-base-turkish-cased", config=config, weights=class_weights)

    args = TrainingArguments(
        output_dir="berturk_tr_idioms_finetuned",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    model.save_pretrained("berturk_tr_idioms_finetuned")
    tokenizer.save_pretrained("berturk_tr_idioms_finetuned")


# Fine-tunes Musixmatch/umberto-commoncrawl-cased-v1
def train_umberto(df):
    label_to_id = {l: i for i, l in enumerate(LABELS)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    df = df[df["language"] == "it"]
    train_df, val_df = train_test_split(df, test_size=0.1)

    def prepare_dataset(data):
        return Dataset.from_dict({
            "tokens": data["tokenized_sentence"].tolist(),
            "labels": [get_bio_tags(row["tokenized_sentence"], row["indices"]) for _, row in data.iterrows()]
        })

    tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")
    train_ds = prepare_dataset(train_df)
    val_ds = prepare_dataset(val_df)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=True)
        tokenized["labels"] = [[label_to_id[label[word_id]] if word_id is not None else -100 for word_id in tokenized.word_ids(i)] for i, label in enumerate(examples["labels"])]
        return tokenized

    train_tok = train_ds.map(tokenize_and_align_labels, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_and_align_labels, batched=True, remove_columns=val_ds.column_names)

    class CustomUmBERTo(AutoModelForTokenClassification):
        def __init__(self, config):
            super().__init__(config)
            self.class_weights = torch.tensor([1.0, 3.0, 2.0])

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device), ignore_index=-100)
                loss = loss_fct(outputs.logits.view(-1, self.config.num_labels), labels.view(-1))
                return {"loss": loss, "logits": outputs.logits}
            return outputs

    config = AutoConfig.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1", num_labels=len(LABELS), id2label=id_to_label, label2id=label_to_id)
    model = CustomUmBERTo.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1", config=config)

    args = TrainingArguments(
        output_dir="umberto_it_idioms_finetuned",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    model.save_pretrained("umberto_it_idioms_finetuned")
    tokenizer.save_pretrained("umberto_it_idioms_finetuned")


# Train trigger function
def train(train_csv_path="train.csv"):

    df = pd.read_csv(train_csv_path)
    df["tokenized_sentence"] = df["tokenized_sentence"].apply(literal_eval)
    df["indices"] = df["indices"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    print("Training XLM-R...")
    train_xlmr(df)

    print("Training BERTurk...")
    train_berturk(df)

    print("Training UmBERTo...")
    train_umberto(df)
