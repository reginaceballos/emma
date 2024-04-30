from data_preparation import response_matrix, clean_responses, selected_questions, label_array

from sklearn import naive_bayes, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import csv
import sys
import matplotlib.pyplot as plt
from tkinter import messagebox
import numpy as np
from sklearn.metrics import roc_curve, auc

def make_bayes_model(combined_responses, label_array):
    ## Make training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_responses, label_array, test_size=0.20, random_state=120)

    ## Count Vectorization
    # create a count vectorizer object
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    vectorizer.fit_transform(X_train).toarray()
    # transform the training and validation data using count vectorizer object
    xtrain_count = vectorizer.transform(X_train)
    xtest_count = vectorizer.transform(X_test)

    model_NB = naive_bayes.MultinomialNB()
    flat_y_train = [item for row in y_train for item in row]
    model_NB.fit(xtrain_count, flat_y_train)
    y_pred_proba = model_NB.predict_proba(xtest_count)[:, 1]

    # predictions = model_NB.predict(xtest_count)
    # flat_y_test = [item for row in y_test for item in row]
    # return (model_NB, predictions, flat_y_test, vectorizer)
    return (y_test, y_pred_proba)

def prepare_new_datapoint(input_file_name, questions_kept, vectorizer):
    input_file_name = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/" + input_file_name
    response_dict = {} # dictionary where the question index is the key and the response is the value
    with open(input_file_name, 'r', encoding='utf-8-sig') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if len(row) > 0:  # Accounts for empty rows in the csv
                [question_index, response_string] = row
                # make response string all lowercase and add to the dictionary
                response_dict[int(question_index)] = response_string.lower()

    # Make a 1 x Q response matrix (list of lists), where Q is the number of questions
    # Make the order of the response matrix match the questions used in the training data
    response_list = []
    for col in range(len(questions_kept)):
        question_index = int(questions_kept[col])
        reponse_to_add = response_dict[question_index]
        response_list.append(reponse_to_add)
    # Add another column to the end (to match the fact that the original matrix had the interview ID in the last column
    response_list.append("New Datapoint")
    response_list_of_lists = [response_list]
    # print(response_list_of_lists)
    cleaned_new_point = clean_responses(response_list_of_lists)

    # Put the responses in the same format as the training dataset
    # Iterate through the cleaned responses to only pull out the columns that correspond to the selected questions
    num_cols = len(cleaned_new_point[0])
    combined_new_point = []
    for c in range(num_cols - 1):  # Leave out the last column with the interview ID
        combined_new_point += cleaned_new_point[0][c]
    new_point_count = vectorizer.transform([combined_new_point])
    return new_point_count

def show_result(prediction):
    # Example computation
    prediction_val = prediction[0]
    if prediction_val == 0:
        result = "not showing signs of depression"
    else:
        result = "showing signs of depression"
    # Show the result in a popup
    messagebox.showinfo("Result", f"User is {result}")

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     input_file_name = sys.argv[1]
    # else:
    #     sys.exit("No input file name provided. Please provide a file name as an argument.")

    ## Prepare the data
    (interview_responses, reversed_question_dict) = response_matrix()
    cleaned_responses = clean_responses(interview_responses)
    [responses_w_IDs, combined_responses, questions_kept] = selected_questions(cleaned_responses)
    label_array = label_array(responses_w_IDs)

    # Make the model
    # (model_NB, predictions, flat_y_test, vectorizer) = make_bayes_model(combined_responses, label_array)
    (y_test, y_pred_proba) = make_bayes_model(combined_responses, label_array)

    # Calculate and Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Depression Detection')
    plt.legend()
    plt.show()




