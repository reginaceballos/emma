# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
# import torch.nn.functional as F
# from umap import UMAP
# from sklearn import preprocessing
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import datasets
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from transformers import AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import Trainer

hf_dataset = load_dataset("dair-ai/emotion", download_mode="force_redownload")
hf_dataset['train'].features['label']
hf_dataset.set_format(type="pandas")
def label_int2str(row):
    return hf_dataset["train"].features["label"].int2str(row)
df_train = hf_dataset["train"][:]
df_test = hf_dataset["test"][:]
df_validation = hf_dataset["validation"][:]

df_train["label_str"] = df_train["label"].apply(label_int2str)
df_test["label_str"] = df_test["label"].apply(label_int2str)
df_validation["label_str"] = df_validation["label"].apply(label_int2str)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

df_train = Dataset.from_pandas(df_train)
df_test = Dataset.from_pandas(df_test)

df_encoded_train = df_train.map(tokenize, batched=True, batch_size=None)
df_encoded_test = df_test.map(tokenize, batched=True, batch_size=None)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

def encode_dataset(df_source):
  df_encoded = df_source.map(tokenize, batched=True, batch_size=None)
  df_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
  return df_encoded

hf_dataset.reset_format()
df_encoded = encode_dataset(hf_dataset)
print(df_encoded)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

def measure_model_performance(trainer, df_input, y_labels, labels_list):
    preds_output = trainer.predict(df_input)
    print(preds_output.metrics)
    y_preds = np.argmax(preds_output.predictions, axis=1)
    plot_confusion_matrix(y_preds, y_labels, labels_list)

def train_model(model_ckpt, training_args, df_encoded_src, num_labels):
    # Create Model
    model = (AutoModelForSequenceClassification
            .from_pretrained(model_ckpt, num_labels=num_labels)
            .to(device))
    model_name = f"models/{model_ckpt}-finetuned-turkish-tweets"
    training_args.output_dir = model_name
    trainer = Trainer(model=model, args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=df_encoded_src["train"],
                    eval_dataset=df_encoded_src["validation"],
                    tokenizer=tokenizer)
    trainer.train()
    return trainer

# Set Batch Size
batch_size = 64
logging_steps = len(df_encoded['train']) // batch_size
num_train_epochs = 2
lr_initial = 2e-5
weight_decay = 1e-3
output_dir = ""
training_args = TrainingArguments(output_dir=output_dir,
                                num_train_epochs=num_train_epochs,
                                learning_rate=lr_initial,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=weight_decay,
                                evaluation_strategy="epoch",
                                disable_tqdm=False,
                                logging_steps=logging_steps,
                                push_to_hub=False,
                                log_level="error")

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
trainer = train_model(model_ckpt, training_args, df_encoded, num_labels=hf_dataset['train'].features['label'].num_classes)
