from data_preparation import response_matrix, clean_responses, selected_questions, label_array

from sklearn import naive_bayes, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import csv
import sys
import matplotlib.pyplot as plt
from tkinter import messagebox
from imblearn.over_sampling import RandomOverSampler
import numpy as np

def make_bayes_model(combined_responses, label_array):
    ## Make training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_responses, label_array, test_size=0.20, random_state=30)

    ## Count Vectorization
    # create a count vectorizer object
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    vectorizer.fit_transform(X_train).toarray()
    # transform the training and validation data using count vectorizer object
    xtrain_count = vectorizer.transform(X_train)
    xtest_count = vectorizer.transform(X_test)

    # Oversample
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=123)
    array_responses = xtrain_count.toarray()
    array_labels = np.array(y_train)
    X_resampled, y_resampled = oversampler.fit_resample(array_responses, array_labels)

    model_NB = naive_bayes.MultinomialNB()
    # flat_y_train = [item for row in y_train for item in row]
    model_NB.fit(X_resampled, y_resampled)
    # model_NB.fit(xtrain_count, flat_y_train)
    predictions = model_NB.predict(xtest_count)
    flat_y_test = [item for row in y_test for item in row]

    ## Evaluate the model
    accuracy = metrics.accuracy_score(predictions, flat_y_test)
    # Make a confusion matrix for the model
    confusion_matrix = metrics.confusion_matrix(flat_y_test, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])

    return (model_NB, accuracy, cm_display, vectorizer)

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
    (model_NB, accuracy, cm_display, vectorizer) = make_bayes_model(combined_responses, label_array)
    print("Accuracy: ", accuracy)
    # Show the confusion matrix
    cm_display.plot()
    plt.show()
    # prepare new datapoint
    cleaned_new_point = prepare_new_datapoint(input_file_name, questions_kept, vectorizer)
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





