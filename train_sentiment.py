# train_sentiment.py

import os
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
# from huggingface_hub import snapshot_download # دیگر نیازی به این نیست

def main(training_strategy: str):
    """
    Main function to load dataset (from local CSV files), tokenize, load model, 
    apply training strategy, train, and evaluate the model for sentiment analysis.
    """
    # --- 1. Load Dataset from Google Drive CSVs and Tokenizer ---
    # مسیر فایل‌های CSV دیتاست در Google Drive شما
    # مطمئن شوید که این مسیرها صحیح است و فایل‌های CSV شما در Google Drive آنجا قرار دارند.
    # فرض می‌کنیم فایل‌ها در پوشه 'nlp_project_data' در MyDrive شما هستند.
    # اگر پوشه دیگری ساختید، مسیر را اصلاح کنید.
    data_folder_path = "/content/drive/MyDrive/nlp_project_data" 
    train_csv_path = os.path.join(data_folder_path, "train.csv")
    dev_csv_path = os.path.join(data_folder_path, "dev.csv") # dev.csv به عنوان validation استفاده می‌شود
    test_csv_path = os.path.join(data_folder_path, "test.csv")
    
    print(f"Loading datasets from local CSV files:")
    print(f"  Train: {train_csv_path}")
    print(f"  Dev (Validation): {dev_csv_path}")
    print(f"  Test:  {test_csv_path}")

    try:
        # خواندن هر سه فایل CSV با pandas
        train_df = pd.read_csv(train_csv_path)
        dev_df = pd.read_csv(dev_csv_path)
        test_df = pd.read_csv(test_csv_path)
        
        # تبدیل DataFrame ها به Hugging Face DatasetDict
        # نام ستون‌های حاوی متن و لیبل در فایل‌های CSV شما باید 'text' و 'label' باشند.
        # اگر نام ستون‌ها متفاوت است، قبل از Dataset.from_pandas باید آن‌ها را تغییر نام دهید.
        # مثال: train_df = train_df.rename(columns={'Your_Text_Column': 'text', 'Your_Label_Column': 'label'})
        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(dev_df), 
            'test': Dataset.from_pandas(test_df)
        })
        
        print("All datasets loaded successfully from local CSV files.")
        print(dataset) # چاپ ساختار دیتاست برای اطمینان
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: One of the CSV files not found: {e}.")
        print(f"Please ensure all train.csv, dev.csv, and test.csv files are uploaded to your Google Drive in the specified path: {data_folder_path}")
        raise # متوقف کردن اجرا اگر فایل پیدا نشد
    except Exception as e:
        print(f"FATAL ERROR: Could not load datasets from CSV files: {e}")
        print("Please check CSV file formats and column names ('text' and 'label').")
        raise # متوقف کردن اجرا اگر خطای دیگری در بارگذاری رخ داد

    model_name = "microsoft/mdeberta-v3-base"
    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")

    # --- 2. Define Tokenization Function and Preprocess Dataset ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("Defining tokenization function.")
    
    print("Tokenizing dataset...")
    # Map tokenization function over all splits (train, validation, test)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # Remove unnecessary columns and rename 'label' column to 'labels' (required by Hugging Face Trainer)
    # '__index_level_0__' ستونی است که pandas ممکن است در صورت reset_index ایجاد کند.
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "__index_level_0__"]) 
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    # Set the format to PyTorch tensors for compatibility with the model
    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"] # استفاده از dev.csv که به validation تغییر نام داده شد
    test_dataset = tokenized_dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. Load Model for Sequence Classification ---
    num_labels = train_df['label'].nunique() # تعداد برچسب‌ها را از DataFrame اصلی بگیرید
    print(f"Number of labels detected: {num_labels}")

    id2label = {0: "negative", 1: "neutral", 2: "positive"} # مطمئن شوید این mapping با داده های شما مطابقت دارد
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