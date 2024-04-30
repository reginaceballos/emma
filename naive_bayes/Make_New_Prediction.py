from data_preparation import clean_responses
import sys
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
def prepare_new_datapoint(input_file_name, saved_questions, vectorizer_list):
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file_name = sys.argv[1]
    else:
        sys.exit("No input file name provided. Please provide a file name as an argument.")

    # Load the model, question list, and vectorizers list that's needed for the prediction
    model_NB = load_pickle_file('naive_bayes_model.pkl')
    vectorizer_list = load_pickle_file('vectorizers.pkl')
    saved_questions = load_pickle_file('saved_questions.pkl')

    # prepare new datapoint
    (transcript, point_final) = prepare_new_datapoint(input_file_name, saved_questions, vectorizer_list)
    # Make a prediction for a new datapoint
    prediction = model_NB.predict(point_final)
    print("Interview transcript: ", transcript)
    print("Prediction: ", prediction)