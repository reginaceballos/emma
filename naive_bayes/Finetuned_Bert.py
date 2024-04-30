from data_preparation import response_matrix, clean_responses, selected_questions, label_array

from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
# import pytorch

## Get training sets and testing sets
def make_train_test_sets(responses_w_IDs, label_array):
    # Remove the interview ID from each response row
    responses = [sublist[1:] for sublist in responses_w_IDs]
    # Make training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(responses, label_array, test_size=0.20, random_state=120)

    # Convert training and test sets into a dataset format that the BERT model can process
    # datasets should start as dictionaries with the following layout:
    # "text": ["example text 1", "example text 2", "testing 3"],
    # "label": [0, 1, 100]  # Assuming binary classification
    training_text_list = []
    training_label_list = []
    num_training_examples = len(X_train)
    question_selected = 0 # Just looking at the first question for now
    for i in range(num_training_examples):
        training_text_list.append(X_train[i][question_selected])
        training_label_list.append(y_train[i][0])
    test_text_list = []
    test_label_list = []
    num_test_examples = len(X_test)
    for i in range(num_test_examples):
        test_text_list.append(X_test[i][question_selected])
        test_label_list.append(y_test[i][0])

    # Create dictionaries using the lists
    train_dictionary = {
        "text": training_text_list,
        "label": training_label_list }
    test_dictionary = {
        "text": test_text_list,
        "label": test_label_list }

    # Create Dataset objects
    training_dataset = Dataset.from_dict(train_dictionary)
    test_dataset = Dataset.from_dict(test_dictionary)

    return (training_dataset, test_dataset)

## Create a preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

## Pass your predictions and labels to compute to calculate the accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def train_model(training_dataset, test_dataset):

    ## Apply the tokenizing function over the entire dataset
    training_tokens = training_dataset.map(preprocess_function, batched=True)
    test_tokens = test_dataset.map(preprocess_function, batched=True)

    ## Dynamically pad the sentences to the longest length in a batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ## Create a map of the expected ids to their labels with id2label and label2id
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    ## Load DistilBERT with AutoModelForSequenceClassification along with the number of expected labels, and the label mappings
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

    ## Define training hyperparameters
    training_args = TrainingArguments(
        output_dir="my_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True )

    ## Pass the training arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= training_tokens,
        eval_dataset= test_tokens,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    ## Call train() to finetune your model
    ## Trainer applies dynamic padding by default when you pass tokenizer to it. In this case, you don’t need to specify a data collator explicitly.
    trainer.train()
    model.save_pretrained("my_model")
    tokenizer.save_pretrained("my_model")

    return 1

def classify():

    ## Once the model is finetuned, use it for inference:
    text = "I'm really sad today. I haven't been getting a lot of sleep."

    # ## Instantiate a pipeline for sentiment analysis with your model, and pass your text to it:
    # classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    # # Use the classifier on your example text
    # print(classifier(text))

    model = AutoModelForSequenceClassification.from_pretrained("my_model")
    tokenizer_new = AutoTokenizer.from_pretrained("my_model")
    inputs = tokenizer_new(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply softmax to logits to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_id = probs.argmax().item()
    confidence = probs.max().item()

    predicted_label = model.config.id2label[predicted_class_id]
    print(f'Predicted class: {predicted_label}, Confidence: {confidence:.4f}')

    return 1

if __name__ == "__main__":

    ## Prepare the data
    (interview_responses, reversed_question_dict) = response_matrix()
    (cleaned_responses, straight_responses) = clean_responses(interview_responses)
    [responses_w_IDs, combined_responses, questions_kept] = selected_questions(straight_responses)
    label_array = label_array(responses_w_IDs)

    # Make the training and testing datasets
    (training_dataset, test_dataset) = make_train_test_sets(responses_w_IDs, label_array)

    # Make new model
    train_model(training_dataset, test_dataset)

    # Classify the data using DistilBERT
    classify()




