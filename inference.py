import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from config import LABELS
from datasets import Dataset

# This function generates predictions for a given df by fitting it to a pretrained model 
def predict_indices(df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    id_to_label = {i: l for i, l in enumerate(LABELS)}

    tokens = df["tokenized_sentence"].tolist()
    dataset = Dataset.from_dict({"tokens": tokens})

    # Tokenize each sentence 
    def tokenize(input):
        return tokenizer(input["tokens"], is_split_into_words=True, truncation=True, padding=True)

    tokenized = dataset.map(tokenize)

    # Uses Trainer to generate predictions 
    # Takes the argmax across logits to get the predicted label ID for each token.
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(tokenized).predictions
    preds = np.argmax(predictions, axis=2)

    # Map predictions back to the original words
    results = []
    for idx in range(len(tokens)):
        word_ids = tokenizer(
            [tokens[idx]], is_split_into_words=True,
            truncation=True, padding=True
        ).word_ids(batch_index=0)

        pred_labels = preds[idx]
        indices = []
        seen = set()
        
        # Collect idiom indices
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in seen:
                seen.add(word_idx)
                if id_to_label[pred_labels[token_idx]] in ["B", "I"]:
                    indices.append(word_idx)

        results.append(indices if indices else [-1])
    return results
