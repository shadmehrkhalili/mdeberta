# train_sentiment.py

import os
import numpy as np
import torch # برای کار با پارامترهای مدل
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate # برای بارگذاری معیارهای ارزیابی
from huggingface_hub import snapshot_download # اضافه کردن این import

def main(training_strategy: str): # اضافه کردن آرگومان training_strategy برای انتخاب استراتژی آموزش
    # --- 1. بارگذاری دیتاست و توکنایزر ---
    dataset_name = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
    print(f"Loading dataset: {dataset_name}")
    
    # *** حل مشکل FileNotFoundError دیتاست در Colab ***
    # این بلاک تلاش می‌کند دیتاست را به صورت مستقیم از Hugging Face Hub لود کند.
    # اگر شکست خورد (که شما FileNotFoundError می‌گرفتید)، آن را به صورت دستی دانلود کرده و از مسیر محلی لود می‌کند.
    try:
        print("Attempting to load dataset from Hugging Face Hub directly...")
        dataset = load_dataset(dataset_name)
        print("Dataset loaded successfully from Hugging Face Hub.")
    except Exception as e: # از Exception کلی استفاده می‌کنیم تا هر نوع خطایی رو بگیریم
        print(f"Direct load from Hugging Face Hub failed: {e}. Attempting manual download and local load...")
        try:
            # این تابع دیتاست را در مسیر کش Hugging Face شما دانلود می کند.
            # اگر دیتاست قبلا دانلود شده باشد، دوباره دانلود نمی کند.
            local_dataset_path = snapshot_download(repo_id=dataset_name, repo_type="dataset")
            print(f"Dataset downloaded to local path: {local_dataset_path}")
            # حالا از مسیر محلی لود می کنیم
            dataset = load_dataset(local_dataset_path)
            print("Dataset loaded successfully from local path.")
        except Exception as download_e:
            print(f"FATAL ERROR: Could not download or load dataset even with manual download: {download_e}")
            raise # اگر دانلود دستی هم مشکل داشت، خطا را دوباره پرتاب می کنیم.

    model_name = "microsoft/mdeberta-v3-base"
    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")

    # --- 2. تعریف تابع پیش‌پردازش (توکنایز کردن) ---
    def tokenize_function(examples):
        # 'text' نام ستون حاوی متن در این دیتاست است.
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("Defining tokenization function.")
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    # حذف ستون‌های غیرضروری و تغییر نام ستون 'label' به 'labels'
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    # تنظیم فرمت خروجی به PyTorch tensors (این برای استفاده توسط مدل‌های ترنسفورمر ضروری است)
    tokenized_dataset.set_format("torch")

    # تقسیم دیتاست به مجموعه آموزش، اعتبارسنجی و تست
    # دیتاست HooshvareLab خودش دارای split های train, validation, test هست.
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. بارگذاری مدل ---
    # تعداد برچسب‌ها رو از دیتاست استخراج می‌کنیم (0: negative, 1: neutral, 2: positive)
    num_labels = dataset["train"].features["label"].num_classes
    print(f"Number of labels detected: {num_labels}")

    # تعیین ID برچسب‌ها (مهم برای فهمیدن خروجی مدل)
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    print(f"Loading model: {model_name} with {num_labels} labels for sequence classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # مدل رو با این IDها آپدیت می‌کنیم (اختیاری اما خوبه برای push کردن به هاب)
    model.config.id2label = id2label
    model.config.label2id = label2id
    print("Model loaded successfully.")
    
    # --- 4. تعریف معیارهای ارزیابی ---
    print("Defining compute_metrics function...")
    # بارگذاری معیارهای مورد نیاز
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # تبدیل لاگیت‌ها به پیش‌بینی‌های نهایی (اندیس کلاس با بالاترین احتمال)
        predictions = np.argmax(logits, axis=-1)
        
        # محاسبه Accuracy
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        
        # محاسبه F1-score، Precision و Recall با average='weighted'
        # برای دسته‌بندی چندکلاسه نامتوازن، از average='weighted' استفاده می‌کنیم
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

    # --- 5. اعمال استراتژی آموزش (فریز کردن پارامترها و تنظیم هایپرپارامترها) ---
    print(f"Applying training strategy: {training_strategy}")
    if training_strategy == "head_only":
        # **استراتژی 1: آموزش فقط Head**
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Model backbone (feature extractor) frozen. Only classification head will be trained.")
        
        num_epochs = 10 
        batch_size = 32 
        learning_rate = 5e-4 

    elif training_strategy == "layer_wise":
        # **استراتژی 2: آموزش Head و لایه‌های آخر Backbone**
        # mDeBERTa V3 Base دارای 12 لایه ترنسفورمر (indexer از 0 تا 11) است.
        # فرض می‌کنیم می‌خواهیم 4 لایه آخر (لایه‌های 8، 9، 10، 11) و Head را آموزش دهیم.
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
        # **استراتژی 3: آموزش کامل مدل (Full Fine-tuning)**
        for name, param in model.named_parameters():
            param.requires_grad = True 
        print("Full model fine-tuning. All parameters will be trained.")
        
        num_epochs = 3
        batch_size = 8
        learning_rate = 5e-6 

    else:
        raise ValueError(f"Unknown training strategy: {training_strategy}. Choose from 'head_only', 'layer_wise', 'full_fine_tune'.")

    # --- تعداد پارامترهای قابل آموزش را چاپ کنید (برای تأیید) ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # --- 6. تنظیم Training Arguments و Trainer ---
    print("Configuring TrainingArguments...")
    # مسیر ذخیره‌سازی مدل در Google Drive شما
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
        # push_to_hub=True, 
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
    
    # --- 7. شروع آموزش و ارزیابی نهایی ---
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset) 
    print(f"\nTest results: {test_results}")

    # ذخیره مدل فاین‌تیون شده نهایی
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")

    # اگر push_to_hub=True رو فعال کرده باشید:
    # trainer.push_to_hub()
    print("Fine-tuning process completed successfully!")

# این بخش برای اجرای اسکریپت با آرگومان از خط فرمان است
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune mDeBERTa-v3-base for sentiment analysis.")
    parser.add_argument("--strategy", type=str, default="head_only",
                        choices=["head_only", "layer_wise", "full_fine_tune"],
                        help="Training strategy: 'head_only', 'layer_wise', or 'full_fine_tune'")
    args = parser.parse_args()
    
    main(training_strategy=args.strategy)