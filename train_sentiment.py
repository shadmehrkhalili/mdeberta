# train_sentiment.py

import os
import numpy as np
import torch
import pandas as pd # Used for reading CSV files
from datasets import Dataset, DatasetDict # Used for creating Hugging Face Dataset objects
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate # For loading evaluation metrics
from pandas.api.types import is_string_dtype, is_integer_dtype # For type checking DataFrame columns
import csv # Required for csv.QUOTE_NONE in pandas read_csv
import argparse # برای خواندن آرگومان‌های خط فرمان

def main():
    """
    Main function to load dataset (from local CSV files), tokenize, load model, 
    apply a specific training strategy, train, and evaluate the model for sentiment analysis.
    """
    # --- خواندن آرگومان‌های ورودی از خط فرمان ---
    parser = argparse.ArgumentParser(description="Fine-tune mDeBERTa-v3-base for sentiment analysis.")
    parser.add_argument("--strategy", type=str, default="head_only",
                        choices=["head_only", "layer_wise", "full_fine_tune"],
                        help="Training strategy to use.")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/mdeberta-v3-base",
                        help="Path to pre-trained model or a previously saved checkpoint.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from. Use 'True' to resume from latest.")
    args = parser.parse_args()
    
    training_strategy = args.strategy
    model_name_or_path = args.model_name_or_path
    resume_checkpoint = args.resume_from_checkpoint

    # --- 1. Load Dataset from Google Drive CSVs and Initialize Tokenizer ---
    # Define paths to your CSV dataset files in Google Drive.
    data_folder_path = "/content/drive/MyDrive/nlp_project_data" 
    train_csv_path = os.path.join(data_folder_path, "train.csv")
    dev_csv_path = os.path.join(data_folder_path, "dev.csv") # dev.csv will be used as validation set
    test_csv_path = os.path.join(data_folder_path, "test.csv")
    
    print(f"Loading datasets from local CSV files:")
    print(f"  Train: {train_csv_path}")
    print(f"  Dev (Validation): {dev_csv_path}")
    print(f"  Test:  {test_csv_path}")

    try:
        # Define expected column names based on the structure of your CSV sample data.
        column_names = ['id', 'text', 'sentiment_str', 'label'] 

        # --- CRITICAL FIX FOR CSV LOADING WITH PANDAS ---
        # This section uses robust pandas.read_csv settings to handle the specific format of your CSVs.
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

        # Verify initial DataFrame load and inspect columns
        print("Initial DataFrame head (train_df):")
        print(train_df.head())
        print("Initial DataFrame columns (train_df):", train_df.columns.tolist())

        # Drop the original string sentiment column ('sentiment_str') as we only need 'text' and 'label' (numeric).
        if 'sentiment_str' in train_df.columns:
            train_df = train_df.drop(columns=['sentiment_str'])
            dev_df = dev_df.drop(columns=['sentiment_str'])
            test_df = test_df.drop(columns=['sentiment_str'])
            print("Dropped 'sentiment_str' column from DataFrames.")

        # Ensure 'text' column is of string type and 'label' is of integer type.
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

        # Convert Pandas DataFrames into Hugging Face DatasetDict format.
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
        print("Please check CSV file formats and the delimiter settings in the script. Ensure no hidden characters or formatting issues exist.")
        print(f"Specific error: {e}")
        raise

    # --- Load Pre-trained Model Tokenizer ---
    print(f"Loading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("Tokenizer loaded successfully.")

    # --- 2. Define Tokenization Function and Preprocess Dataset ---
    def tokenize_function(examples):
        # Tokenizes the text column, truncating to max_length and padding to the longest sequence in the batch.
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("Defining tokenization function.")
    
    print("Tokenizing dataset...")
    # Apply the tokenization function to all splits of the dataset (train, validation, test)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # Remove unnecessary columns from the tokenized dataset.
    cols_to_remove = ["id"] 
    if "__index_level_0__" in tokenized_dataset['train'].column_names:
        cols_to_remove.append("__index_level_0__")
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in cols_to_remove if col in tokenized_dataset['train'].column_names])
    
    # Rename the 'label' column to 'labels' as required by Hugging Face Trainer.
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels") 
    # Set the format of the dataset to PyTorch tensors for compatibility with PyTorch models and Trainer.
    tokenized_dataset.set_format("torch")

    # Assign the prepared datasets to variables for Trainer
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. Load Model for Sequence Classification ---
    # Determine the number of labels dynamically from the 'label' column of the training DataFrame.
    num_labels = train_df['label'].nunique() 
    print(f"Unique labels found in training data: {sorted(train_dataset.unique('labels'))}")
    print(f"Number of labels detected: {num_labels}")

    # Define ID to label and label to ID mappings based on the detected number of labels.
    if num_labels == 2: 
        id2label = {0: "HAPPY", 1: "SAD"} 
        label2id = {"HAPPY": 0, "SAD": 1}
        print("Using 2-label mapping (HAPPY, SAD).")
    else: # Fallback for any other number of labels
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        label2id = {f"LABEL_{i}": i for i in range(num_labels)}
        print(f"Warning: Auto-generated id2label and label2id for {num_labels} labels.")

    print(f"Loading model: {model_name_or_path} with {num_labels} labels for sequence classification")
    # Load the pre-trained model. Pass id2label and label2id directly for better configuration.
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels,
                                                               id2label=id2label, label2id=label2id)
    print("Model loaded successfully.")
    
    # --- 4. Define Evaluation Metrics (Accuracy, F1, Precision, Recall) ---
    # **FIX:** Define f1_average_method in the main scope to be accessible by both functions.
    f1_average_method = "weighted" if num_labels > 2 else "binary"

    print("Defining compute_metrics function...")
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        
        # Use the f1_average_method variable defined in the outer scope
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
        layers_to_unfreeze = 4 
        num_total_layers = getattr(model.config, 'num_hidden_layers', 12) 
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
        raise ValueError(f"Unknown training strategy: {training_strategy}.")

    # --- Print Trainable Parameters (for confirmation) ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # --- 6. Configure Training Arguments and Trainer ---
    print("Configuring TrainingArguments...")
    output_dir = f"/content/drive/MyDrive/mdeberta_sentiment_{training_strategy}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

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
        eval_strategy="epoch", # Corrected parameter name
        save_strategy="epoch",              
        load_best_model_at_end=True,        
        metric_for_best_model=f"f1_{f1_average_method}" if num_labels > 1 else "accuracy", # FIX: Use variable from main scope
        greater_is_better=True,             
        report_to="none", # Disable wandb logging to avoid prompts
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
    
    # --- 7. Start Training and Final Evaluation ---
    print("Starting model training...")
    
    # Check if we should resume from a checkpoint
    if resume_checkpoint:
        # If 'True', resume from the latest checkpoint in output_dir. Otherwise, use the provided path.
        resume_from_cp = True if resume_checkpoint.lower() == 'true' else resume_checkpoint
        print(f"Resuming training from checkpoint: {resume_from_cp}")
        trainer.train(resume_from_checkpoint=resume_from_cp)
    else:
        trainer.train()
        
    print("Training finished.")

    print("\nEvaluating the BEST model on the TEST set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"\nTest results: {test_results}")

    # Save the final (best) model and its tokenizer
    final_model_path = os.path.join(output_dir, "final_model_and_tokenizer")
    trainer.save_model(final_model_path)
    print(f"Fine-tuned model and tokenizer saved to {final_model_path}")

    print("Fine-tuning process completed successfully!")

if __name__ == "__main__":
    main()