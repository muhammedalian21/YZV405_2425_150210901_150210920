# Idiom Identification Pipeline

This repository contains a complete pipeline for **training** and **evaluating** idiom identification models using transformer-based architectures (XLM-R, BERTurk, UmBERTo).

---

## 🚀 Usage

All functionality is managed through `main.py`.

### 📦 1. Setup

Install required dependencies:

``` bash
pip install -r requirements.txt
``` 

Download pretrained model weights from the provided drive link and **unzip them into the project directory** so that the following folders exist:

🔗 **Download pretrained models:**  
[Download from Google Drive](https://drive.google.com/file/d/1l6r0fznJYVk4-gTD6Z4rKT7wx_dKDxDO/view?usp=drive_link)

``` text
./xlmr_large_idiom_finetuned/
./berturk_tr_idioms_finetuned/
./umberto_it_idioms_finetuned/
``` 

---


### 🧠 2. Inference (Evaluation)

To run inference on a dataset and output predictions:

``` bash
python main.py \
    --eval_csv path/to/eval.csv \
    --output_csv path/to/output.csv
``` 

Optional: You can override the model paths:

``` bash
    --turkish_model path/to/berturk_model \
    --italian_model path/to/umberto_model \
    --xlmr_model path/to/xlmr_model
``` 

---

### 🔁 3. Training (Optional)

To **train all models from scratch** using the training CSV:

``` bash
python main.py --train --train_csv path/to/train.csv
``` 

If no path is specified, it defaults to `data/train.csv` as defined in `config.py`.

---

## 📄 File Format Specifications

### 📥 Input File (`train.csv` / `eval.csv`)

| Column Name         | Description                                    |
|---------------------|------------------------------------------------|
| `id`                | Unique sentence ID                            |
| `language`          | Language code: `tr` or `it`                    |
| `sentence`          | Original sentence (not used in training)       |
| `tokenized_sentence`| List of word tokens   |
| `expression`        | Idiomatic expression              |
| `category`          | Idiom category                   |
| `indices`           | List of indices marking idiom in sentence     |

> `indices` is `[-1]` if no idiom is present.  
> `tokenized_sentence` and `indices` must be stringified lists (e.g. `"['this', 'is', 'a', 'test']"`).


---

## ⚙️ Configuration

All default paths and label mappings are defined in `config.py`. You can override them using command-line arguments.

---

## ✅ Assumptions

- The input tokenized sentences and idiom indices are preprocessed.
- Model weights must be available locally (downloaded & unzipped from provided link).
- Only Turkish (`tr`) and Italian (`it`) languages are supported.
- You don’t need to train if you use the pretrained weights.

---

## 👨‍💻 Example Run

``` bash
python main.py \
    --eval_csv eval_w-o_labels.csv \
    --output_csv eval_predictions.csv
``` 

