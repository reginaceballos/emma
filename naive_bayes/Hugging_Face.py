from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

## Load data
imdb = load_dataset("imdb")

## Pre-process
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

## Create a preprocessing function to tokenize text and truncate sequences to be no longer
## than DistilBERT’s maximum input length
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

## Apply the preprocessing function over the entire dataset
## Speed up map by setting batched=True to process multiple elements of the dataset at once
imdb = {split: imdb[split].map(preprocess_function, batched=True) for split in imdb.keys()}

# Now you can continue with selecting the range and the rest of your process.
# Cut the dataset down to just the first 200 rows after tokenization to maintain Dataset object structure
tokenized_imdb = {split: imdb[split].select(range(200)) for split in imdb.keys()}

## Now create a batch of examples using DataCollatorWithPadding
## Dynamically pad the sentences to the longest length in a batch
## (Rather than padding the whole dataset to the maximum length)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

## Pass your predictions and labels to compute to calculate the accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

## Create a map of the expected ids to their labels with id2label and label2id
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

## Load DistilBERT with AutoModelForSequenceClassification along with the number of expected labels, and the label mappings
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

## Define your training hyperparameters in TrainingArguments
## Only required parameter is output_dir which specifies where to save your model.
## At the end of each epoch, the Trainer will evaluate the accuracy and save the training checkpoint

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False, ## Changed this to False, b/c "ush this model to the Hub by setting push_to_hub=True (you need to be signed in to Hugging Face to upload your model)"
)

## Pass the training arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

## Call train() to finetune your model
## Trainer applies dynamic padding by default when you pass tokenizer to it. In this case, you don’t need to specify a data collator explicitly.
trainer.train()

## Once the model is finetuned, use it for inference:
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

## Instantiate a pipeline for sentiment analysis with your model, and pass your text to it:
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Use the classifier on your example text
print(classifier(text))
