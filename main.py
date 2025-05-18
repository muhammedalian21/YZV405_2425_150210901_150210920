import argparse
import pandas as pd
from config import *
from data import load_dataset
from inference import predict_indices
from train import train


def main(args):

    if args.train:
        train(train_csv_path=args.train_csv)
        return

    df = load_dataset(args.input_csv)

    df_tr = df[df["language"] == "tr"].copy()
    df_it = df[df["language"] == "it"].copy()

    if not df_tr.empty:
        df_tr["indices"] = predict_indices(df_tr, args.turkish_model)
    if not df_it.empty:
        df_it["indices"] = predict_indices(df_it, args.italian_model)

    result = pd.concat([df_tr, df_it], ignore_index=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, action="store_true", help="If set, trains all models from scratch")
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV, help="Path to training file")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV, help="Path to input file")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to output predictions file")
    parser.add_argument("--xlmr_model", type=str, default=DEFAULT_XLMR_MODEL, help="Path to the XLM-R model")
    parser.add_argument("--turkish_model", type=str, default=DEFAULT_TURKISH_MODEL, help="Path to the Turkish model")
    parser.add_argument("--italian_model", type=str, default=DEFAULT_ITALIAN_MODEL, help="Path to the Italian model")

    args = parser.parse_args()
    main(args)
