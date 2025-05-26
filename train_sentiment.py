# train_sentiment.py

import os
import numpy as np
import torch # برای کار با پارامترهای مدل
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate # برای بارگذاری معیارهای ارزیابی

def main(training_strategy: str): # اضافه کردن آرگومان training_strategy برای انتخاب استراتژی آموزش
    # --- 1. بارگذاری دیتاست و توکنایزر ---
    dataset_name = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print("Dataset loaded successfully.")

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
    
    # بقیه کد در مراحل بعدی اضافه خواهند شد...
    print("Dataset preprocessed and model loaded. Ready for defining strategy logic and metrics.")

# این بخش برای اجرای اسکریپت با آرگومان از خط فرمان است
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune mDeBERTa-v3-base for sentiment analysis.")
    parser.add_argument("--strategy", type=str, default="head_only",
                        choices=["head_only", "layer_wise", "full_fine_tune"],
                        help="Training strategy: 'head_only', 'layer_wise', or 'full_fine_tune'")
    args = parser.parse_args()
    
    main(training_strategy=args.strategy)

    # ... (previous code in main() function, from stage 2) ...

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

    # بقیه کد در مراحل بعدی اضافه خواهند شد...
    print("Metrics defined. Ready for strategy application and Trainer setup.")

# ... (rest of the file) ...