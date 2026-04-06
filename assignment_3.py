"""this is the last assignment of Natural language proccessing
this was made by group 11 comprised of:
Miruna Lungu (S5882206)
Andrejs Tupikins (S5607442)
Prayer Aguebor (S5901782)
"""

import re
import html
import evaluate
import torch as pt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
)

target_names = ["World", "Sports", "Business", "Sci/Tech"]

news_dataset = load_dataset("ag_news")
full_dataset = concatenate_datasets([news_dataset["train"], news_dataset["test"]])

split = full_dataset.train_test_split(test_size=0.3, seed=42)
train_data = split["train"]
val_test_data = split["test"]
test_valid = val_test_data.train_test_split(test_size=0.5, seed=42)

""" data split into 70% train 15% validation 15% test """

val_data = test_valid["train"]
test_data = test_valid["test"]

# training data
train_text = train_data["text"]
train_label = train_data["label"]

# validation data
val_text = val_data["text"]
val_label = val_data["label"]

# used at the end for testing
test_text = test_data["text"]
test_label = test_data["label"]


def preProcessing(text):
    text = re.sub(r"(?<!&)#(\d+);", r"&#\1;", text)
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_text = [preProcessing(text) for text in train_text]
val_text = [preProcessing(text) for text in val_text]
test_text = [preProcessing(text) for text in test_text]

checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)

train_dataset = Dataset.from_dict({"text": train_text, "label": train_label})
val_dataset = Dataset.from_dict({"text": val_text, "label": val_label})
test_dataset = Dataset.from_dict({"text": test_text, "label": test_label})


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)


train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./results",
    eval_steps=2000,
    save_steps=2000,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=500,
    logging_strategy="steps",
    save_total_limit=2,
    seed=42,
    report_to="none",
)

metric_accuracy = evaluate.load("accuracy")
metric_f1_score = evaluate.load("f1")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1_score.compute(
        predictions=predictions, references=labels, average="macro"
    )
    precision = metric_precision.compute(
        predictions=predictions, references=labels, average="macro"
    )
    recall = metric_recall.compute(
        predictions=predictions, references=labels, average="macro"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "macro_f1": f1["f1"],
        "macro_precision": precision["precision"],
        "macro_recall": recall["recall"],
    }


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
early_stop = EarlyStoppingCallback(early_stopping_patience=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = DistilBertForSequenceClassification.from_pretrained(
    checkpoint, num_labels=4
).to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stop],
    data_collator=data_collator,
)

trainer.train()

# Evaluation
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = np.array(test_label)

print(
    classification_report(
        true_labels, predicted_labels, target_names=target_names, digits=4
    )
)

test_accuracy = accuracy_score(true_labels, predicted_labels)
test_macro_f1 = f1_score(true_labels, predicted_labels, average="macro")

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro-F1: {test_macro_f1:.4f}")

ConfusionMatrixDisplay.from_predictions(
    true_labels, predicted_labels, display_labels=target_names, cmap="Blues"
)
plt.title("DistilBERT Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()

# Error Analysis
mis_idx = [i for i in range(len(true_labels)) if true_labels[i] != predicted_labels[i]]
misclassified_examples_bert = []

for i in mis_idx[:20]:
    misclassified_examples_bert.append(
        {
            "text": test_text[i],
            "true": target_names[true_labels[i]],
            "pred": target_names[predicted_labels[i]],
        }
    )

df_errors = pd.DataFrame(misclassified_examples_bert)
print(df_errors.head(20))
df_errors.to_csv("misclassified_examples.csv", index=False)

# Length Bucket Analysis
lengths = [len(text.split()) for text in test_text]


def get_bucket(length):
    if length <= 15:
        return "0-15"
    elif length <= 30:
        return "16-30"
    elif length <= 60:
        return "31-60"
    else:
        return "61+"


buckets = [get_bucket(length) for length in lengths]
length_results = []

for bucket in ["0-15", "16-30", "31-60", "61+"]:
    bucket_idx = [i for i in range(len(buckets)) if buckets[i] == bucket]
    if len(bucket_idx) == 0:
        continue
    bucket_true = np.array([true_labels[i] for i in bucket_idx])
    bucket_pred = np.array([predicted_labels[i] for i in bucket_idx])
    length_results.append(
        {
            "bucket": bucket,
            "n": len(bucket_idx),
            "accuracy": accuracy_score(bucket_true, bucket_pred),
            "macro_f1": f1_score(bucket_true, bucket_pred, average="macro"),
        }
    )

df_length = pd.DataFrame(length_results)
print(df_length)
df_length.to_csv("length_bucket_results.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(df_length["bucket"], df_length["accuracy"], marker="o", label="Accuracy")
plt.plot(df_length["bucket"], df_length["macro_f1"], marker="o", label="Macro-F1")
plt.title("Performance by Input Length Bucket")
plt.xlabel("Length Bucket")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.savefig("length_bucket_results.png", bbox_inches="tight")
plt.show()

# Key Word Masking
class_keywords = {
    "World": [
        "government",
        "president",
        "minister",
        "country",
        "election",
        "iraq",
        "china",
        "russia",
        "israel",
        "united nations",
    ],
    "Sports": [
        "game",
        "team",
        "season",
        "coach",
        "player",
        "match",
        "league",
        "cup",
        "tournament",
        "score",
    ],
    "Business": [
        "market",
        "stocks",
        "shares",
        "company",
        "profit",
        "revenue",
        "investor",
        "bank",
        "economy",
        "sales",
        "deal",
    ],
    "Sci/Tech": [
        "software",
        "technology",
        "internet",
        "computer",
        "chip",
        "microsoft",
        "google",
        "data",
        "network",
        "science",
        "space",
    ],
}


def mask_keywords(text, keywords):
    masked_text = text
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = r"\b" + re.escape(keyword) + r"\b"
        masked_text = re.sub(
            pattern, tokenizer.mask_token, masked_text, flags=re.IGNORECASE
        )
    return re.sub(r"\s+", " ", masked_text).strip()


masked_test_text = [
    mask_keywords(text, class_keywords[target_names[label]])
    for text, label in zip(test_text, test_label)
]

masked_test_dataset = Dataset.from_dict({"text": masked_test_text, "label": test_label})
masked_test_dataset = masked_test_dataset.map(preprocess_function, batched=True)
masked_test_dataset = masked_test_dataset.rename_column("label", "labels")
masked_test_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

masked_predictions = trainer.predict(masked_test_dataset)
masked_predicted_labels = np.argmax(masked_predictions.predictions, axis=1)

masked_acc = accuracy_score(true_labels, masked_predicted_labels)
masked_f1 = f1_score(true_labels, masked_predicted_labels, average="macro")

masking_results = pd.DataFrame(
    [
        {"condition": "Original", "accuracy": test_accuracy, "macro_f1": test_macro_f1},
        {"condition": "Masked", "accuracy": masked_acc, "macro_f1": masked_f1},
        {
            "condition": "Drop",
            "accuracy": test_accuracy - masked_acc,
            "macro_f1": test_macro_f1 - masked_f1,
        },
    ]
)

print(masking_results)
masking_results.to_csv("keyword_masking_results.csv", index=False)

plt.figure(figsize=(7, 5))
plt.plot(
    ["Original", "Masked"], [test_accuracy, masked_acc], marker="o", label="Accuracy"
)
plt.plot(
    ["Original", "Masked"], [test_macro_f1, masked_f1], marker="o", label="Macro-F1"
)
plt.title("Keyword Masking Probe")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.savefig("keyword_masking_results.png", bbox_inches="tight")
plt.show()
