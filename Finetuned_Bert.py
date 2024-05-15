from data_preparation import response_matrix, clean_responses, selected_questions, label_array

from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline, BertForSequenceClassification, EarlyStoppingCallback

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix

## Get training sets and testing sets
def make_train_test_sets(responses_w_IDs, label_array, questions_kept):
    # Remove the interview ID from each response row
    responses = [sublist[1:] for sublist in responses_w_IDs]
    # Make training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(responses, label_array, test_size=0.20, random_state= 182)

    # Convert training and test sets into a dataset format that the BERT model can process
    # datasets should start as dictionaries with the following layout:
    # "text": ["example text 1", "example text 2", "testing 3"],
    # "label": [0, 1, 100]  # Assuming binary classification

    # # Training with just one question
    # question_selected = 4 # Just looking at the first question for now
    # num_training_examples = len(X_train)
    # training_text_list = []
    # training_label_list = []
    # for i in range(num_training_examples):
    #     training_text_list.append(X_train[i][question_selected])
    #     training_label_list.append(y_train[i][0])
    # test_text_list = []
    # test_label_list = []
    # num_test_examples = len(X_test)
    # for i in range(num_test_examples):
    #     test_text_list.append(X_test[i][question_selected])
    #     test_label_list.append(y_test[i][0])
    #     print(X_test[i][question_selected])

    # Training with multiple questions
    question_selected_1 = 0  # Just looking at the first question for now
    question_selected_2 = 3
    # question_selected_3 = 2
    num_training_examples = len(X_train)
    training_text_list = []
    training_label_list = []
    for i in range(num_training_examples):
        answer_1 = X_train[i][question_selected_1]
        answer_2 = X_train[i][question_selected_2]
        # answer_3 = X_train[i][question_selected_3]
        # training_text_list.append(answer_1 + " [SEP] " + answer_2 + " [SEP] " + answer_3)
        training_text_list.append(answer_1 + " [SEP] " + answer_2)
        training_label_list.append(y_train[i][0])
    test_text_list = []
    test_label_list = []
    num_test_examples = len(X_test)
    for i in range(num_test_examples):
        answer_1 = X_test[i][question_selected_1]
        answer_2 = X_test[i][question_selected_2]
        # answer_3 = X_test[i][question_selected_3]
        # test_text_list.append(answer_1 + " [SEP] " + answer_2 + " [SEP] " + answer_3)
        test_text_list.append(answer_1 + " [SEP] " + answer_2)
        test_label_list.append(y_test[i][0])
        # print(X_test[i][question_selected])

    # ## Training on all questions, separating each question by a SEP indicator
    # training_text_list = []
    # training_label_list = []
    # for response, label in zip(X_train, y_train):
    #     # Concatenate responses using a separator
    #     concatenated_response = " [SEP] ".join(response[i] for i in range(len(questions_kept)))
    #     training_text_list.append(concatenated_response)
    #     training_label_list.append(label[0])
    # test_text_list = []
    # test_label_list = []
    # for response, label in zip(X_test, y_test):
    #     concatenated_response = " [SEP] ".join(response[i] for i in range(len(questions_kept)))
    #     test_text_list.append(concatenated_response)
    #     test_label_list.append(label[0])

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
    print(test_dictionary)

    return (training_dataset, test_dataset)

## Create a preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

## Pass your predictions and labels to compute to calculate the accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    print("Confusion Matrix:\n", conf_matrix)

    return accuracy.compute(predictions=predictions, references=labels)

def make_tokens(training_dataset, test_dataset):
    ## Apply the tokenizing function over the entire dataset
    training_tokens = training_dataset.map(preprocess_function, batched=True)
    test_tokens = test_dataset.map(preprocess_function, batched=True)
    return (training_tokens, test_tokens)

def train_model(training_tokens, test_tokens, model_name):

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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement after 3 evaluations
    )

    ## Call train() to finetune your model
    ## Trainer applies dynamic padding by default when you pass tokenizer to it. In this case, you don’t need to specify a data collator explicitly.
    trainer.train()
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)

    return 1


## Once the model is finetuned, use it for inference:
def classify_new(text, model_name):

    # ## Instantiate a pipeline for sentiment analysis with your model, and pass your text to it:
    # classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    # # Use the classifier on your example text
    # print(classifier(text))

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer_new = AutoTokenizer.from_pretrained(model_name)
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

# Compute the probability scores for the positive class after training.
def get_model_predictions(test_tokens, model, tokenizer):
    trainer = Trainer(model=model, tokenizer=tokenizer)
    raw_pred, _, _ = trainer.predict(test_tokens)
    # Assuming the second column contains probabilities for the positive class
    probabilities = torch.nn.functional.softmax(torch.from_numpy(raw_pred), dim=-1)[:,1].numpy()
    return probabilities

# After training the model and obtaining the probabilities, compute the ROC curve
def plot_roc_curve(test_labels, probabilities):
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def adjust_threshold(test_tokens, model, tokenizer, new_threshold):
    # Predict probabilities for the positive class
    probabilities = get_model_predictions(test_tokens, model, tokenizer)
    # Adjust the threshold
    predictions = (probabilities > new_threshold).astype(int)

    # Get true labels from your test dataset
    test_labels = test_dataset['label']
    test_labels_array = np.array(test_labels)

    # Calculate confusion matrix and other metrics
    conf_matrix = confusion_matrix(test_labels_array, predictions)
    print("Lower Threshold Confusion Matrix:\n", conf_matrix)

    # Calculate additional metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    # Compute metrics
    acc_score = accuracy.compute(predictions=predictions, references=test_labels_array)
    prec_score = precision.compute(predictions=predictions, references=test_labels_array)
    rec_score = recall.compute(predictions=predictions, references=test_labels_array)
    f1_score = f1.compute(predictions=predictions, references=test_labels_array)

    print("Lower Threshold Accuracy:", acc_score)
    print("Lower Threshold Precision:", prec_score)
    print("Lower Threshold Recall:", rec_score)
    print("Lower Threshold F1 Score:", f1_score)

    # Plot ROC curve
    plot_roc_curve(test_labels_array, probabilities)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("Provide a name for the model")

    ## Prepare the data
    (interview_responses, reversed_question_dict) = response_matrix()
    (cleaned_responses, straight_responses) = clean_responses(interview_responses)
    [responses_w_IDs, combined_responses, questions_kept] = selected_questions(straight_responses)
    label_array = label_array(responses_w_IDs)

    print("Questions kept: ", questions_kept)

    # Make the training and testing datasets
    (training_dataset, test_dataset) = make_train_test_sets(responses_w_IDs, label_array, questions_kept)
    (training_tokens, test_tokens) = make_tokens(training_dataset, test_dataset)

    ####### Train the model new model #######
    train_model(training_tokens, test_tokens, model_name)

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # Find the accuracy
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )
    # results = trainer.evaluate(eval_dataset=test_tokens)
    # print("Accuracy:", results['eval_accuracy'])

    # ## Plot the ROC curve
    # # Load the model (Being done above)
    # # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # # Predict probabilities for the test set
    # probabilities = get_model_predictions(test_tokens, model, tokenizer)
    # # Get true labels from your test dataset
    # test_labels = test_dataset['label']
    # test_labels_array = np.array(test_labels)
    # # Plot ROC curve
    # plot_roc_curve(test_labels, probabilities)
    #
    # # Classify the data using DistilBERT
    # text = "i feel i feel i feel nothing but love I can feel nothing"
    # classify_new(text, model_name)

    # Try using a new threshold
    new_threshold = 0.35
    adjust_threshold(test_tokens, model, tokenizer, new_threshold)
