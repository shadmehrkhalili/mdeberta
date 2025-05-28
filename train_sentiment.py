# train_sentiment.py

import os
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from pandas.api.types import is_string_dtype, is_integer_dtype
import csv

def main(training_strategy: str):
    """
    Main function to load dataset (from local CSV files), tokenize, load model,
    apply training strategy, train, and evaluate the model for sentiment analysis.
    """
    # --- 1. Load Dataset from Google Drive CSVs and Tokenizer ---
    data_folder_path = "/content/drive/MyDrive/nlp_project_data"
    train_csv_path = os.path.join(data_folder_path, "train.csv")
    dev_csv_path = os.path.join(data_folder_path, "dev.csv") # dev.csv will be used as validation
    test_csv_path = os.path.join(data_folder_path, "test.csv")

    print(f"Loading datasets from local CSV files:")
    print(f"  Train: {train_csv_path}")
    print(f"  Dev (Validation): {dev_csv_path}")
    print(f"  Test:  {test_csv_path}")

    try:
        column_names = ['id', 'text', 'sentiment_str', 'label']

        train_df = pd.read_csv(train_csv_path, sep='\t', header=None, names=column_names,
                               skiprows=1,
                               quotechar=None, quoting=csv.QUOTE_NONE,
                               engine='c', on_bad_lines='skip',
                               encoding='utf-8')
        dev_df = pd.read_csv(dev_csv_path, sep='\t', header=None, names=column_names,
                             skiprows=1,
                             quotechar=None, quoting=csv.QUOTE_NONE,
                             engine='c', on_bad_lines='skip',
                             encoding='utf-8')
        test_df = pd.read_csv(test_csv_path, sep='\t', header=None, names=column_names,
                              skiprows=1,
                              quotechar=None, quoting=csv.QUOTE_NONE,
                              engine='c', on_bad_lines='skip',
                              encoding='utf-8')

        print("Successfully loaded CSVs with explicit tab delimiter, C parser, and header skipping.")
        print("Initial DataFrame head (train_df):")
        print(train_df.head())
        print("Initial DataFrame columns (train_df):", train_df.columns.tolist())

        if 'sentiment_str' in train_df.columns:
            train_df = train_df.drop(columns=['sentiment_str'])
            dev_df = dev_df.drop(columns=['sentiment_str'])
            test_df = test_df.drop(columns=['sentiment_str'])
            print("Dropped 'sentiment_str' column from DataFrames.")

        if 'text' in train_df.columns and not train_df['text'].isnull().all():
            if not is_string_dtype(train_df['text']):
                train_df['text'] = train_df['text'].astype(str)
                dev_df['text'] = dev_df['text'].astype(str)
                test_df['text'] = test_df['text'].astype(str)
                print("Converted 'text' column to string type.")
        elif 'text' not in train_df.columns or train_df['text'].isnull().all():
            raise ValueError("Critical Error: 'text' column is missing or all NaN after loading. Check CSV format and content.")

        if 'label' in train_df.columns and not train_df['label'].isnull().all():
            if not is_integer_dtype(train_df['label']):
                train_df['label'] = train_df['label'].astype(int)
                dev_df['label'] = dev_df['label'].astype(int)
                test_df['label'] = test_df['label'].astype(int)
                print("Converted 'label' column to integer type.")
        elif 'label' not in train_df.columns or train_df['label'].isnull().all():
            raise ValueError("Critical Error: 'label' column is missing or all NaN after loading. Cannot convert to int. Check CSV format and content.")

        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(dev_df),
            'test': Dataset.from_pandas(test_df)
        })
        print("All datasets loaded successfully from local CSV files and converted to DatasetDict.")
        print(dataset)

    except FileNotFoundError as e:
        print(f"FATAL ERROR: One of the CSV files not found: {e}.")
        print(f"Please ensure all train.csv, dev.csv, and test.csv files are uploaded to your Google Drive in the specified path: {data_folder_path}")
        raise
    except Exception as e:
        print(f"FATAL ERROR: Could not load datasets from CSV files: {e}")
        print("Please check CSV file formats and the delimiter settings in the script.")
        print(f"Specific error: {e}")
        raise

    model_name = "microsoft/mdeberta-v3-base"
    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("Defining tokenization function.")
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    cols_to_remove = ["id"]
    if "__index_level_0__" in tokenized_dataset['train'].column_names:
        cols_to_remove.append("__index_level_0__")
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in cols_to_remove if col in tokenized_dataset['train'].column_names])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if 'labels' in train_dataset.features:
        unique_labels = train_dataset.unique("labels")
        num_labels = len(unique_labels)
        print(f"Unique labels found in training data: {unique_labels}")
    else:
        raise ValueError("Could not determine number of labels. 'labels' column missing in tokenized training data.")
    print(f"Number of labels detected: {num_labels}")

    f1_average_method = "weighted" if num_labels > 2 else "binary"

    if num_labels == 3:
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        label2id = {"negative": 0, "neutral": 1, "positive": 2}
    elif num_labels == 2:
        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}
    else:
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        label2id = {f"LABEL_{i}": i for i in range(num_labels)}
        print(f"Warning: Auto-generated id2label and label2id for {num_labels} labels.")

    print(f"Loading model: {model_name} with {num_labels} labels for sequence classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                               id2label=id2label, label2id=label2id)
    print("Model loaded successfully.")

    print("Defining compute_metrics function...")
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average=f1_average_method)
        precision = precision_metric.compute(predictions=predictions, references=labels, average=f1_average_method)
        recall = recall_metric.compute(predictions=predictions, references=labels, average=f1_average_method)
        return {
            "accuracy": accuracy["accuracy"],
            f"f1_{f1_average_method}": f1["f1"],
            f"precision_{f1_average_method}": precision["precision"],
            f"recall_{f1_average_method}": recall["recall"],
        }
    print("compute_metrics function defined.")

    print(f"Applying training strategy: {training_strategy}")
    if training_strategy == "head_only":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Model backbone (feature extractor) frozen. Only classification head will be trained.")
        num_epochs = 10
        batch_size = 32
        learning_rate = 5e-4
    elif training_strategy == "layer_wise":
        layers_to_unfreeze = 4
        if hasattr(model.config, 'num_hidden_layers'):
            num_total_layers = model.config.num_hidden_layers
        else:
            num_total_layers = 12
            print(f"Warning: model.config.num_hidden_layers not found. Assuming {num_total_layers} layers for layer_wise unfreezing.")
        for name, param in model.named_parameters():
            if "classifier" in name or \
               any(f"encoder.layer.{i}" in name for i in range(num_total_layers - layers_to_unfreeze, num_total_layers)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"Model backbone (last {layers_to_unfreeze} layers) and classification head will be trained.")
        num_epochs = 6
        batch_size = 16
        learning_rate = 2e-5
    elif training_strategy == "full_fine_tune":
        for name, param in model.named_parameters():
            param.requires_grad = True
        print("Full model fine-tuning. All parameters will be trained.")
        num_epochs = 3
        batch_size = 8
        learning_rate = 5e-6
    else:
        raise ValueError(f"Unknown training strategy: {training_strategy}. Choose from 'head_only', 'layer_wise', 'full_fine_tune'.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    print("Configuring TrainingArguments...")
    output_dir = f"/content/drive/MyDrive/mdeberta_sentiment_{training_strategy}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    metric_to_optimize_for_best_model = f"f1_{f1_average_method}" if num_labels > 1 else "accuracy"

    if len(train_dataset) > 0 and batch_size > 0 and num_epochs > 0:
        total_training_steps = (len(train_dataset) // batch_size) * num_epochs
        warmup_ratio = 0.1
        calculated_warmup_steps = int(total_training_steps * warmup_ratio)
    else:
        calculated_warmup_steps = 500

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=max(100, calculated_warmup_steps),
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_strategy="epoch",
        eval_strategy="epoch",  # <--- CORRECTED ARGUMENT NAME
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_to_optimize_for_best_model,
        greater_is_better=True,
    )
    print("TrainingArguments configured.")

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    print("Trainer initialized.")

    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"\nTest results: {test_results}")

    final_model_path = os.path.join(output_dir, "final_model_and_tokenizer")
    trainer.save_model(final_model_path)
    print(f"Fine-tuned model and tokenizer saved to {final_model_path}")

    print("Fine-tuning process completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune mDeBERTa-v3-base for sentiment analysis.")
    parser.add_argument("--strategy", type=str, default="head_only",
                        choices=["head_only", "layer_wise", "full_fine_tune"],
                        help="Training strategy: 'head_only', 'layer_wise', or 'full_fine_tune'")
    args = parser.parse_args()

    main(training_strategy=args.strategy)