from data_preparation import response_matrix, clean_responses, selected_questions, label_array

from sklearn import naive_bayes, metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
import csv
import sys
import matplotlib.pyplot as plt
from tkinter import messagebox
import numpy as np
from sklearn.metrics import roc_curve, auc
import pickle
import evaluate

# Helper function for lamda in make bayes model
def identity_analyzer(x):
    return x

## Make a pre-trained Bayes model
def make_bayes_model(responses_w_IDs, label_array):
    # Remove the interview ID from each response row
    responses = [sublist[1:] for sublist in responses_w_IDs]
    indices = np.arange(len(responses))  # Generate indices
    # Splitting data and indices to make training and testing sets
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(responses, label_array, indices, test_size=0.20, random_state=20)

    # ## Make training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(responses, label_array, test_size=0.20, random_state= 20)
    # X_train, X_test, y_train, y_test = train_test_split(responses, label_array, test_size=0.20)

    ## Count Vectorization
    # For each question asked by EMMA, make a new vectorizer for that question
    num_questions = len(X_train[0])
    vectorizer_list = [] # save the vectorizers as a list so that they can be used on the new datapoint later
    train_features_list = []
    test_features_list = []
    for i in range(num_questions):
        one_q_list_train = [sublist[i] for sublist in X_train]
        one_q_list_test = [sublist[i] for sublist in X_test]
        # print(one_q_list_test)
        # print("===================================================================")
        # print()
        # create a count vectorizer object
        # vectorizer = CountVectorizer(analyzer=lambda x: x) # Had a problem saving this as a pickle object
        vectorizer = CountVectorizer(analyzer=identity_analyzer)
        vectorizer.fit_transform(one_q_list_train).toarray()
        # transform the training and validation data using count vectorizer object
        xtrain_count = vectorizer.transform(one_q_list_train)
        xtest_count = vectorizer.transform(one_q_list_test)
        train_features_list.append(xtrain_count.toarray())
        test_features_list.append(xtest_count.toarray())
        vectorizer_list.append(vectorizer)

    # Concatenate all the vectorized arays into one feature array
    X_train_final = np.hstack(train_features_list)
    X_test_final = np.hstack(test_features_list)

    model_NB = naive_bayes.MultinomialNB()
    flat_y_train = [item for row in y_train for item in row]
    model_NB.fit(X_train_final, flat_y_train)
    y_pred_proba = model_NB.predict_proba(X_test_final)[:, 1] # for ROC curve
    predictions = model_NB.predict(X_test_final)
    flat_y_test = [item for row in y_test for item in row]

    return (model_NB, predictions, flat_y_test, vectorizer_list, y_pred_proba, y_test, X_test)

## Prepare a new data point to be tested by the model
def prepare_new_datapoint(input_file_name, questions_kept, vectorizer_list, input_file_given):
    if input_file_given == 1:
        input_file_name = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/" + input_file_name
        response_dict = {} # dictionary where the question index is the key and the response is the value
        with open(input_file_name, 'r', encoding='utf-8-sig') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                if len(row) > 0:  # Accounts for empty rows in the csv
                    [question_index, response_string] = row
                    # make response string all lowercase and add to the dictionary
                    response_dict[int(question_index)] = response_string.lower()
    else: # When there's no input file provided, make a fake input file
        response_dict = {}  # dictionary where the question index is the key and the response is the value
        pre_made_dict = {}
        pre_made_dict[1] = "i'm doing well today thanks for asking"
        pre_made_dict[4] = "the last time i was really happy was yesterday when i spent too day with my dog and my family we went to the park"
        pre_made_dict[5] = "i got into an argument with my mom last month because she wants me to come home for christmas, but i told her i'm going to my partner's house for christmas"
        pre_made_dict[6] = "i have been sleeping pretty easily i usually get a minimum oof eight hours a night"
        pre_made_dict[30] = "i don't really feel guilty about anything maybe on procrastinating on my homework"
        for q in questions_kept:
            if q in pre_made_dict:
                response_dict[q] = pre_made_dict[q]
            else:
                response_dict[q] = "test_response"

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
    (cleaned_new_point, straight_point) = clean_responses(response_list_of_lists)

    ### Vectorize the new point
    # # For each question asked by EMMA, apply the vectorizer for that question
    num_questions = len(cleaned_new_point[0]) - 1 #subtract 1 to ignore the last column with the interview ID
    point_features_list = []
    for i in range(num_questions):
        vectorizer = vectorizer_list[i]
        one_q_list_point = [sublist[i] for sublist in cleaned_new_point]
        point_count = vectorizer.transform(one_q_list_point)
        point_features_list.append(point_count.toarray())

    # Concatenate the vectorized arrays into one feature array
    point_final = np.hstack(point_features_list)

    return point_final

## Calculate cross validation metrics
def calc_cross_validation(responses_w_IDs, label_array):
    # Remove the interview ID from each response row
    responses = [sublist[1:] for sublist in responses_w_IDs]
    ## Count Vectorization
    num_questions = len(responses[0])

    transformed_responses = []
    for i in range(num_questions):
        one_q_list = [sublist[i] for sublist in responses]
        vectorizer = CountVectorizer(analyzer=lambda x: x)
        one_q_count = vectorizer.fit_transform(one_q_list)
        transformed_responses.append(one_q_count.toarray())

    response_count = np.hstack(transformed_responses)
    model_CV = naive_bayes.MultinomialNB()
    flat_label_array = [item for row in label_array for item in row]

    return (model_CV, response_count, flat_label_array)

## Show the depression result as a pop-up
def show_result(prediction):
    # Example computation
    prediction_val = prediction[0]
    if prediction_val == 0:
        result = "not showing signs of depression"
    else:
        result = "showing signs of depression"
    # Show the result in a popup
    messagebox.showinfo("Result", f"User is {result}")

## Save the pre-trained model so it can be used later
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

## Save the vectorizers list so it can be used later
def save_vectorizers(vectorizers, filename):
    with open(filename, 'wb') as file:
        pickle.dump(vectorizers, file)

## Save the questions list so it can be used later
def save_questions(question_list, filename):
    with open(filename, 'wb') as file:
        pickle.dump(question_list, file)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file_name = sys.argv[1]
        input_file_given = 1
    else:
        print("No input file csv file provided for a new prediction. Defaulting to internal test file.", "\n")
        input_file_name = "no input file given"
        input_file_given = 0
        # sys.exit("No input file name provided. Please provide a file name as an argument.")

    ## Prepare the data
    (interview_responses, reversed_question_dict) = response_matrix()
    (cleaned_responses, staight_responses) = clean_responses(interview_responses)
    [responses_w_IDs, combined_responses, questions_kept] = selected_questions(cleaned_responses)
    label_array = label_array(responses_w_IDs)

    # Make the model
    (model_NB, predictions, flat_y_test, vectorizer_list, y_pred_proba, y_test, X_test) = make_bayes_model(responses_w_IDs, label_array)

    # Change the threshold value
    new_threshold = 0.000025  # Threshold of 0.000025, random seed of 20 for the demo

    # Classify instances based on the new threshold
    predictions_lower_threshold = (y_pred_proba >= new_threshold).astype(int)

    # Compute metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    acc_score = accuracy.compute(predictions=predictions_lower_threshold, references=flat_y_test)
    prec_score = precision.compute(predictions=predictions_lower_threshold, references=flat_y_test)
    rec_score = recall.compute(predictions=predictions_lower_threshold, references=flat_y_test)
    f1_score = f1.compute(predictions=predictions_lower_threshold, references=flat_y_test)

    print("Lower Threshold Accuracy: ", acc_score)
    print("Lower Threshold Precision: ", prec_score)
    print("Lower Threshold Recall: ", rec_score)
    print("Lower Threshold F1: ", f1_score)

    # Evaluate performance with the new threshold
    fpr_lower, tpr_lower, thresholds_lower = roc_curve(y_test, y_pred_proba)
    roc_auc_lower = auc(fpr_lower, tpr_lower)

    # Plot the ROC curve with the new threshold
    plt.figure()
    plt.plot(fpr_lower, tpr_lower, label='ROC curve (area = %0.2f)' % roc_auc_lower)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Depression Detection (Lower Threshold)')
    plt.legend()
    plt.show()

    # Evaluate the model using Cross-Validation with the new threshold
    (model_CV, response_count, flat_label_array) = calc_cross_validation(responses_w_IDs, label_array)
    cv_scores_lower_threshold = cross_val_score(model_NB, response_count, flat_label_array, cv=KFold(5, shuffle=True))
    # print("Cross-validated scores with lower threshold:", cv_scores_lower_threshold)
    print("Mean CV Accuracy with lower threshold: ", np.mean(cv_scores_lower_threshold))
    print("==========================================")

    # Make a confusion matrix:
    # Calculate the confusion matrix
    conf_matrix_lower_threshold = metrics.confusion_matrix(flat_y_test, predictions_lower_threshold)
    # Display the confusion matrix
    disp_lower_threshold = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lower_threshold, display_labels=[0, 1])
    disp_lower_threshold.plot()
    plt.title('Confusion Matrix (Lower Threshold)')
    plt.show()

    # prepare new datapoint
    cleaned_new_point = prepare_new_datapoint(input_file_name, questions_kept, vectorizer_list, input_file_given)

    # Make a prediction for a new datapoint
    # prediction = model_NB.predict(cleaned_new_point)
    y_pred_proba = model_NB.predict_proba(cleaned_new_point)[:, 1]  # Get probabilities for the positive class
    prediction = (y_pred_proba >= new_threshold).astype(int)  # Apply the new threshold
    if prediction[0] == 0:
        print("Prediction: Risk of depression not detected")
    elif prediction[0] == 1:
        print("Prediction: Risk of depression detected")
    else:
        print("error: Prediction not found")

    # Save model, question list, and vectorizers list to be integrated in the front end
    save_model(model_NB, 'naive_bayes_model.pkl')
    save_vectorizers(vectorizer_list, 'vectorizers.pkl')
    save_questions(questions_kept, 'saved_questions.pkl')









