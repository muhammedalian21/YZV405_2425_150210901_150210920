import pandas as pd
from ast import literal_eval
from datasets import Dataset

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df["tokenized_sentence"] = df["tokenized_sentence"].apply(literal_eval)
    return df

def tokenize_dataset(df, tokenizer):
    dataset = Dataset.from_dict({
        "tokens": df["tokenized_sentence"].tolist(),
        "language": df["language"].tolist(),
        "id": df["id"].tolist()
    })

    def tokenize(examples):
        return tokenizer(
            examples["tokens"],
            truncation=True,
            padding=True,
            is_split_into_words=True
        )

    return dataset.map(tokenize, batched=True)
