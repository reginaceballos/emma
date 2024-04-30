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

def make_bayes_model(responses_w_IDs, label_array):
    # Remove the interview ID from each response row
    responses = [sublist[1:] for sublist in responses_w_IDs]
    ## Make training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(responses, label_array, test_size=0.20, random_state=120)
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
        # create a count vectorizer object
        vectorizer = CountVectorizer(analyzer=lambda x: x)
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

    return (model_NB, predictions, flat_y_test, vectorizer_list, y_pred_proba, y_test)

def prepare_new_datapoint(input_file_name, questions_kept, vectorizer_list):
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

    # # Put the responses in the same format as the training dataset
    # # Iterate through the cleaned responses to only pull out the columns that correspond to the selected questions
    # num_cols = len(cleaned_new_point[0])
    # combined_new_point = []
    # for c in range(num_cols - 1):  # Leave out the last column with the interview ID
    #     combined_new_point += cleaned_new_point[0][c]
    # new_point_count = vectorizer.transform([combined_new_point])

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
    if len(sys.argv) > 1:
        input_file_name = sys.argv[1]
    else:
        sys.exit("No input file name provided. Please provide a file name as an argument.")

    ## Prepare the data
    (interview_responses, reversed_question_dict) = response_matrix()
    cleaned_responses = clean_responses(interview_responses)
    [responses_w_IDs, combined_responses, questions_kept] = selected_questions(cleaned_responses)
    label_array = label_array(responses_w_IDs)

    # Make the model
    (model_NB, predictions, flat_y_test, vectorizer_list, y_pred_proba, y_test) = make_bayes_model(responses_w_IDs, label_array)

    ## Evaluate the model
    ### Accuracy
    accuracy = metrics.accuracy_score(predictions, flat_y_test)
    print("Accuracy: ", accuracy)
    print("Questions listed: ", questions_kept)
    ### Confusion matrix
    confusion_matrix = metrics.confusion_matrix(flat_y_test, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.show()
    ### ROC curve
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


    # prepare new datapoint
    cleaned_new_point = prepare_new_datapoint(input_file_name, questions_kept, vectorizer_list)
    # Make a prediction for a new datapoint
    prediction = model_NB.predict(cleaned_new_point)
    print("Prediction: ", prediction)

    # ## Show the result in a pop up window
    # # Create the main window
    # root = tk.Tk()
    # root.withdraw()  # Hide the main window as we only want to show the popup
    # # Call the function to show the popup
    # show_result(prediction)
    # # Start the Tkinter event loop
    # root.mainloop()





