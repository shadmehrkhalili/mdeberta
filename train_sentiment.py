# train_sentiment.py

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from huggingface_hub import snapshot_download # مطمئن شوید این import وجود دارد

def main(training_strategy: str):
    """
    Main function to load dataset, tokenize, load model, apply training strategy,
    train, and evaluate the model for sentiment analysis.
    """
    # --- 1. Load Dataset and Tokenizer ---
    dataset_name = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
    print(f"Loading dataset: {dataset_name}")
    
    # *** Robust Dataset Loading Logic to handle FileNotFoundError ***
    # This block attempts to load the dataset directly from Hugging Face Hub.
    # If it fails (e.g., due to caching issues or transient network problems),
    # it then tries to manually download it using snapshot_download and load from local path.
    try:
        print("Attempting to load dataset from Hugging Face Hub directly...")
        dataset = load_dataset(dataset_name)
        print("Dataset loaded successfully from Hugging Face Hub.")
    except Exception as e: # Catching a broad exception to be robust
        print(f"Direct load from Hugging Face Hub failed: {e}. Attempting manual download and local load...")
        try:
            # snapshot_download will cache the dataset locally in Colab's filesystem
            # and won't re-download if it already exists.
            local_dataset_path = snapshot_download(repo_id=dataset_name, repo_type="dataset")
            print(f"Dataset downloaded to local path: {local_dataset_path}")
            # Now load the dataset from the local path
            dataset = load_dataset(local_dataset_path)
            print("Dataset loaded successfully from local path.")
        except Exception as download_e:
            print(f"FATAL ERROR: Could not download or load dataset even with manual download: {download_e}")
            print("Please check your network connection, the dataset ID, or Hugging Face Hub status.")
            raise # Re-raise the exception if even manual download/load fails

    model_name = "microsoft/mdeberta-v3-base"
    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")

    # --- 2. Define Tokenization Function and Preprocess Dataset ---
    def tokenize_function(examples):
        # 'text' is the column name for text in the Snappfood dataset.
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("Defining tokenization function.")
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # Remove unnecessary columns and rename 'label' column to 'labels' (required by Trainer)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    # Set the format to PyTorch tensors for compatibility with the model
    tokenized_dataset.set_format("torch")

    # Split dataset into train, validation, and test sets
    # The HooshvareLab dataset already has these splits.
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. Load Model for Sequence Classification ---
    # Determine the number of labels from the dataset features (0: negative, 1: neutral, 2: positive)
    num_labels = dataset["train"].features["label"].num_classes
    print(f"Number of labels detected: {num_labels}")

    # Define ID to label and label to ID mappings (for clear output and model config)
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    print(f"Loading model: {model_name} with {num_labels} labels for sequence classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Update model config with label mappings
    model.config.id2label = id2label
    model.config.label2id = label2id
    print("Model loaded successfully.")
    
    # --- 4. Define Evaluation Metrics (Accuracy, F1, Precision, Recall) ---
    print("Defining compute_metrics function...")
    # Load required evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert logits to final predictions (index of the class with highest probability)
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate Accuracy
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        
        # Calculate F1-score, Precision, and Recall using 'weighted' averaging for multi-class imbalance
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
        # Strategy 1: Train Only the Classification Head
        # Freeze all model parameters except for the 'classifier' layer (the head).
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
            # else: print(f"Parameter '{name}' is NOT frozen and will be trained (Head).")
        print("Model backbone (feature extractor) frozen. Only classification head will be trained.")
        
        # Optimized hyperparameters for Head-Only training (faster, less resource-intensive)
        num_epochs = 10
        batch_size = 32
        learning_rate = 5e-4 # Higher learning rate for Head-Only training

    elif training_strategy == "layer_wise":
        # Strategy 2: Train Head and Last Layers of Backbone
        # mDeBERTa V3 Base has 12 Transformer layers (indexed 0 to 11).
        # We'll train the last 4 layers (layers 8, 9, 10, 11) and the classification head.
        for name, param in model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(8, 12)) or "classifier" in name:
                param.requires_grad = True # These layers will be trained
                # print(f"Parameter '{name}' is NOT frozen (Layer-wise).")
            else:
                param.requires_grad = False # Remaining layers are frozen
        print("Model backbone (last 4 layers) and classification head will be trained.")
        
        # Optimized hyperparameters for Layer-Wise training (medium resources, good performance)
        num_epochs = 6 
        batch_size = 16
        learning_rate = 2e-5 # Moderate learning rate

    elif training_strategy == "full_fine_tune":
        # Strategy 3: Full Fine-tuning (Train all model parameters)
        for name, param in model.named_parameters():
            param.requires_grad = True # All parameters will be trained
        print("Full model fine-tuning. All parameters will be trained.")
        
        # Optimized hyperparameters for Full Fine-tuning (most resource-intensive, highest risk of overfitting)
        num_epochs = 3
        batch_size = 8
        learning_rate = 5e-6 # Lower learning rate as many parameters are updated

    else:
        raise ValueError(f"Unknown training strategy: {training_strategy}. Choose from 'head_only', 'layer_wise', 'full_fine_tune'.")

    # --- Print Trainable Parameters (for confirmation) ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # --- 6. Configure Training Arguments and Trainer ---
    print("Configuring TrainingArguments...")
    # Output directory in Google Drive (make sure this folder exists or is created by your script)
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
        # push_to_hub=True, # Uncomment and configure if pushing to Hugging Face Hub
        # hub_model_id=f"YOUR_HUGGINGFACE_USERNAME/mdeberta-v3-base-snappfood-{training_strategy}", 
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

    # If push_to_hub=True was enabled in TrainingArguments and logged in:
    # trainer.push_to_hub()
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