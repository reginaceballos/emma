import sys
sys.path.append("/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/GitHub_emma/")  # Replace with the path to the folder containing data_preparation.py
from speech_to_text import speech_to_text

from data_preparation import clean_responses

import pickle
import csv
import numpy as np

## Load files stored as pickles
def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Helper function to read in pickle vectorizer list
def identity_analyzer(x):
    return x

## Prepare a new data point to be tested by the model
def prepare_new_datapoint(saved_questions, vectorizer_list):
    # Make a dictionary where the question index is the key and the response is the value
    response_dict = {}

    # Get a dataframe with the responses to each question
    response_df = speech_to_text()

    for index, row in response_df.iterrows():
        if len(row) >= 2:  # Ensure there are at least two columns
            question_index = row[0]  # first column has the question index
            response_string = row[1]  # second column has the response
            response_dict[int(question_index)] = response_string.lower() # make response string all lowercase and add to the dictionary

    print(saved_questions)

    # Make a 1 x Q response matrix (list of lists), where Q is the number of questions
    # Make the order of the response matrix match the questions used in the training data
    response_list = []
    for col in range(len(saved_questions)):
        question_index = int(saved_questions[col])
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

    return (response_list, point_final)

def make_new_prediction():
    # Load the model, question list, and vectorizers list that's needed for the prediction
    model_NB = load_pickle_file('naive_bayes_model.pkl')
    vectorizer_list = load_pickle_file('vectorizers.pkl')
    saved_questions = load_pickle_file('saved_questions.pkl')

    # prepare new datapoint
    (transcript, point_final) = prepare_new_datapoint(saved_questions, vectorizer_list)

    # Make a prediction for a new datapoint
    prediction = int(model_NB.predict(point_final)[0])

    if prediction == 0:
        string_diagnosis = "Patient is not at risk of depression"
    elif prediction == 1:
        string_diagnosis = "Patient is at risk of depression"
    else:
        string_diagnosis = "Error in generating a prediction"

    confidence = "N/A"

    return string_diagnosis, confidence

if __name__ == "__main__":
    print(make_new_prediction())