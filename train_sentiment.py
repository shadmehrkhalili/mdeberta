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


# ... (previous code in main() function, from stage 3) ...

    # --- 5. اعمال استراتژی آموزش (فریز کردن پارامترها و تنظیم هایپرپارامترها) ---
    print(f"Applying training strategy: {training_strategy}")
    if training_strategy == "head_only":
        # **استراتژی 1: آموزش فقط Head**
        # فریز کردن تمام پارامترهای مدل به جز لایه دسته‌بندی (classifier/head)
        for name, param in model.named_parameters():
            if "classifier" not in name: # "classifier" نام لایه Head در مدل‌های Hugging Face است
                param.requires_grad = False
            # else: # میتونید این خط رو فعال کنید تا ببینید کدوم پارامترها آموزش میبینن
            #     print(f"Parameter '{name}' is NOT frozen and will be trained (Head).")
        print("Model backbone (feature extractor) frozen. Only classification head will be trained.")
        
        # هایپرپارامترهای بهینه برای آموزش فقط Head
        num_epochs = 10 # تعداد اپوک‌ها رو بیشتر کنید چون سریعتره
        batch_size = 32 # اندازه بچ میتونه بزرگتر باشه چون RAM کمتری مصرف میشه
        learning_rate = 5e-4 # نرخ یادگیری بالاتر برای Head

    elif training_strategy == "layer_wise":
        # **استراتژی 2: آموزش Head و لایه‌های آخر Backbone**
        # mDeBERTa V3 Base دارای 12 لایه ترنسفورمر (indexer از 0 تا 11) است.
        # فرض می‌کنیم می‌خواهیم 4 لایه آخر (لایه‌های 8، 9، 10، 11) و Head را آموزش دهیم.
        for name, param in model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(8, 12)) or "classifier" in name:
                param.requires_grad = True # این لایه‌ها آموزش می‌بینند
                # print(f"Parameter '{name}' is NOT frozen (Layer-wise).")
            else:
                param.requires_grad = False # بقیه فریز می‌شوند
        print("Model backbone (last 4 layers) and classification head will be trained.")
        
        # هایپرپارامترهای بهینه برای آموزش لایه‌ای
        num_epochs = 6 # تعداد اپوک‌ها کمتر از Head-Only، بیشتر از Full
        batch_size = 16 # اندازه بچ متوسط
        learning_rate = 2e-5 # نرخ یادگیری متوسط

    elif training_strategy == "full_fine_tune":
        # **استراتژی 3: آموزش کامل مدل (Full Fine-tuning)**
        # تمام پارامترهای مدل آموزش می‌بینند.
        for name, param in model.named_parameters():
            param.requires_grad = True # همه پارامترها آموزش می‌بینند
        print("Full model fine-tuning. All parameters will be trained.")
        
        # هایپرپارامترهای بهینه برای Full Fine-tuning
        num_epochs = 3 # تعداد اپوک‌ها باید کمتر باشد (Overfitting)
        batch_size = 8 # اندازه بچ باید کوچکتر باشد (مصرف RAM بالا)
        learning_rate = 5e-6 # نرخ یادگیری باید پایین‌تر باشد

    else:
        raise ValueError(f"Unknown training strategy: {training_strategy}. Choose from 'head_only', 'layer_wise', 'full_fine_tune'.")

    # --- تعداد پارامترهای قابل آموزش را چاپ کنید (برای تأیید) ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # --- 6. تنظیم Training Arguments و Trainer ---
    print("Configuring TrainingArguments...")
    # مسیر ذخیره‌سازی مدل در Google Drive شما (این پوشه رو خودتون در درایو بسازید)
    output_dir = f"/content/drive/MyDrive/mdeberta_snappfood_sentiment_{training_strategy}"

    # اگر پوشه خروجی وجود نداشت، آن را ایجاد کن (مهم برای Colab)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,        # تعداد اپوک‌ها بر اساس استراتژی
        per_device_train_batch_size=batch_size, # اندازه بچ بر اساس استراتژی
        per_device_eval_batch_size=batch_size,  # اندازه بچ برای ارزیابی
        learning_rate=learning_rate,        # نرخ یادگیری بر اساس استراتژی
        warmup_steps=500,                   # گام‌های warmup
        weight_decay=0.01,                  # وزن‌کاهی (برای جلوگیری از overfitting)
        logging_dir=f'{output_dir}/logs',   # مسیر لاگ‌ها برای TensorBoard
        logging_strategy="epoch",           # لاگ‌برداری بعد از هر اپوک
        evaluation_strategy="epoch",        # ارزیابی بعد از هر اپوک
        save_strategy="epoch",              # ذخیره مدل بعد از هر اپوک
        load_best_model_at_end=True,        # بارگذاری بهترین مدل بر اساس معیار در پایان
        metric_for_best_model="f1_weighted", # معیاری که برای انتخاب بهترین مدل استفاده می‌شود
        greater_is_better=True,             # آیا مقدار بالاتر این معیار بهتر است؟ (بله برای F1)
        # push_to_hub=True, # اگر می‌خواهید مدل رو به Hugging Face Hub آپلود کنید (نیاز به notebook_login)
        # hub_model_id=f"YOUR_HUGGINGFACE_USERNAME/mdeberta-v3-base-snappfood-{training_strategy}", # نام ریپازیتوری در هاب
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
    
    # بقیه کد در مراحل بعدی اضافه خواهند شد...
    print("Trainer is ready. Next step: Training and Evaluation.")

# ... (rest of the file) ...

# ... (previous code in main() function, from stage 4) ...

    # --- 7. شروع آموزش و ارزیابی نهایی ---
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    print("\nEvaluating on test set...")
    # برای ارزیابی نهایی از test_dataset استفاده می‌کنیم (همونطور که در Trainer تعریف شده)
    test_results = trainer.evaluate(eval_dataset=test_dataset) 
    print(f"\nTest results: {test_results}")

    # ذخیره مدل فاین‌تیون شده نهایی
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")

    # اگر push_to_hub=True رو در TrainingArguments فعال کرده باشید و لاگین کرده باشید:
    # trainer.push_to_hub()
    print("Fine-tuning process completed successfully!")

# ... (the `if __name__ == "__main__":` block at the very end of the file) ...