# train_sentiment.py

import os
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from pandas.api.types import is_string_dtype
import csv # برای تنظیمات quoting در pandas

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
        # Reading all three CSV files with pandas using highly robust parameters for tab-separated data
        # 'sep='\t'': Explicitly set tab as separator.
        # 'header=None': Indicate that there is no header row in the CSVs.
        # 'names': Provide column names explicitly. Based on your sample: ID, Text, String Sentiment, Numeric Label.
        # 'quotechar=None', 'quoting=csv.QUOTE_NONE': Prevent pandas from interpreting quotes, assuming fields are unquoted.
        # 'engine='python'': Python engine is slower but more flexible and less prone to C parser errors.
        # 'on_bad_lines='skip'': Skips problematic lines entirely, preventing ParserError.
        # 'lineterminator='\n'': Explicitly define line ending character for robust reading.
        # 'encoding='utf-8'': Ensure proper UTF-8 encoding.
        
        column_names = ['id', 'text', 'sentiment_str', 'label'] # Define expected column names

        train_df = pd.read_csv(train_csv_path, sep='\t', header=None, names=column_names, 
                               quotechar=None, quoting=csv.QUOTE_NONE, 
                               engine='python', on_bad_lines='skip', 
                               lineterminator='\n', encoding='utf-8')
        dev_df = pd.read_csv(dev_csv_path, sep='\t', header=None, names=column_names,
                             quotechar=None, quoting=csv.QUOTE_NONE, 
                             engine='python', on_bad_lines='skip', 
                             lineterminator='\n', encoding='utf-8')
        test_df = pd.read_csv(test_csv_path, sep='\t', header=None, names=column_names,
                              quotechar=None, quoting=csv.QUOTE_NONE, 
                              engine='python', on_bad_lines='skip', 
                              lineterminator='\n', encoding='utf-8')
        
        # Verify initial load and columns (optional but good for debugging)
        print("Initial DataFrame head (train_df):")
        print(train_df.head())
        print("Initial DataFrame columns (train_df):", train_df.columns.tolist())

        # Drop the original string sentiment column as we only need 'text' and 'label' (numeric)
        if 'sentiment_str' in train_df.columns:
            train_df = train_df.drop(columns=['sentiment_str'])
            dev_df = dev_df.drop(columns=['sentiment_str'])
            test_df = test_df.drop(columns=['sentiment_str'])
            print("Dropped 'sentiment_str' column from DataFrames.")

        # Ensure 'text' column is string type and 'label' is integer type
        if not is_string_dtype(train_df['text']):
            train_df['text'] = train_df['text'].astype(str)
            dev_df['text'] = dev_df['text'].astype(str)
            test_df['text'] = test_df['text'].astype(str)
            print("Converted 'text' column to string type.")
        
        if not pd.api.types.is_integer_dtype(train_df['label']):
            train_df['label'] = train_df['label'].astype(int)
            dev_df['label'] = dev_df['label'].astype(int)
            test_df['label'] = test_df['label'].astype(int)
            print("Converted 'label' column to integer type.")


        # Convert DataFrames to Hugging Face DatasetDict
        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(dev_df), 
            'test': Dataset.from_pandas(test_df)
        })
        
        print("All datasets loaded successfully from local CSV files and converted to DatasetDict.")
        print(dataset) # Print dataset structure for confirmation
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: One of the CSV files not found: {e}.")
        print(f"Please ensure all train.csv, dev.csv, and test.csv files are uploaded to your Google Drive in the specified path: {data_folder_path}")
        raise # Stop execution if file not found
    except Exception as e:
        print(f"FATAL ERROR: Could not load datasets from CSV files: {e}")
        print("Please check CSV file formats and the delimiter. It seems to be TAB-separated with no headers.")
        print(f"Specific pandas error: {e}") # Display the exact pandas error for more debugging if needed
        raise # Stop execution if other loading error occurs

    model_name = "microsoft/mdeberta-v3-base"
    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")

    # --- 2. Define Tokenization Function and Preprocess Dataset ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("Defining tokenization function.")
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # Remove unnecessary columns from tokenized dataset for Trainer
    # '__index_level_0__' is an index column pandas might create when converting to Dataset.
    # 'id' column from your CSV also needs to be removed.
    cols_to_remove = ["__index_level_0__", "id"] 
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in cols_to_remove if col in tokenized_dataset['train'].column_names])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. Load Model for Sequence Classification ---
    # Determine the number of labels from the original train_df
    num_labels = train_df['label'].nunique() 
    print(f"Number of labels detected: {num_labels}")

    # Ensure this mapping matches your actual label IDs (0, 1) and their meanings (e.g., negative, positive)
    # The dataset name "bert-fa-base-uncased-sentiment-snappfood" implies 3 sentiments: negative, neutral, positive.
    # And your sample has 0 (HAPPY) and 1 (SAD). This is a potential mismatch.
    # If your labels are ONLY 0 and 1, you should adjust num_labels and id2label/label2id.
    # For now, let's assume 3 labels as per original dataset description.
    # If your CSVs only contain 0 and 1, `train_df['label'].nunique()` will return 2.
    # If it returns 2, you MUST change `id2label` and `label2id` to match only 2 labels.
    # Example: id2label = {0: "negative", 1: "positive"}
    # For now, we assume 3 as per public dataset's intended labels.
    id2label = {0: "negative", 1: "neutral", 2: "positive"} # Adjust if your actual labels are different (e.g., only 0 and 1)
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    print(f"Loading model: {model_name} with {num_labels} labels for sequence classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.id2label = id2label
    model.config.label2id = label2id
    print("Model loaded successfully.")
    
    # --- 4. Define Evaluation Metrics (Accuracy, F1, Precision, Recall) ---
    print("Defining compute_metrics function...")
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        precision_weighted = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
        recall_weighted = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
        
        return {
            "accuracy": accuracy["accuracy"],
            "f1_weighted": f1_weighted["f1"],
            "precision_weighted": precision_weighted["precision"],
            "recall_weighted": recall_weighted["recall"],
        }
    print("compute_metrics function defined.")

    # --- 5. Apply Training Strategy (Freeze parameters and set hyperparameters) ---
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
        for name, param in model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(8, 12)) or "classifier" in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False 
        print("Model backbone (last 4 layers) and classification head will be trained.")
        
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

    # --- 6. Configure Training Arguments and Trainer ---
    print("Configuring TrainingArguments...")
    output_dir = f"/content/drive/MyDrive/mdeberta_snappfood_sentiment_{training_strategy}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,        
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size,  
        learning_rate=learning_rate,        
        warmup_steps=500,                   
        weight_decay=0.01,                  
        logging_dir=f'{output_dir}/logs',   
        logging_strategy="epoch",           
        evaluation_strategy="epoch",        
        save_strategy="epoch",              
        load_best_model_at_end=True,        
        metric_for_best_model="f1_weighted", 
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
    )
    print("Trainer initialized.")
    
    # --- 7. Start Training and Final Evaluation ---
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset) 
    print(f"\nTest results: {test_results}")

    # Save the fine-tuned model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")

    print("Fine-tuning process completed successfully!")

# This block allows the script to be run from the command line with arguments
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune mDeBERTa-v3-base for sentiment analysis.")
    parser.add_argument("--strategy", type=str, default="head_only",
                        choices=["head_only", "layer_wise", "full_fine_tune"],
                        help="Training strategy: 'head_only', 'layer_wise', or 'full_fine_tune'")
    args = parser.parse_args()
    
    main(training_strategy=args.strategy)