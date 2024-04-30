### Overview ###
# Make a matrix where the rows correspond to different users
# And the columns correspond to the questions in EMMA-Question-List.csv

# import libraries
import csv
import os
import copy
import numpy as np
import xlsxwriter
import re


import nltk
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()

## Make a matrix with interviews on the x axis and all the poossible questioons on the y axis
def response_matrix():
    # Determine how many unqiue "Master" questions there are
    question_key_dict = {} # dictionary with master question as key and the index (column number) as the value
    question_map_dict = {} # dictionary with sub question as key and master question as the value
    question_key_dict_reversed = {} # dictionary with the index as the key and the master question as the value
    question_list_csv = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/Full-EMMA-Question-List.csv"
    with open(question_list_csv, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader) #skip first row
        for row in datareader:
            if len(row) > 0:  # Accounts for empty rows in the csv
                [master_question, index, sub_question, frequency] = row
                if master_question not in question_key_dict:
                    question_key_dict[master_question] = int(index)
                    question_key_dict_reversed[int(index)] = master_question

                question_map_dict[sub_question] = master_question

    num_questions = len(question_key_dict.keys())

    # Iterate through each interview and put their responses into matrix spot [user][question]
    directory = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/Transcripts"

    # Find the number of files in the folder
    num_interviews = 0
    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and "DS_Store" not in f:
            num_interviews += 1

    # Make an empty user response matrix where there are num_interviews rows and num_questions + 1 columns
    # (where the extra column 0 is for the introduction before the questions)
    # And the extra column on the end is for the ID of the participant (e.g. 310)
    interview_responses = [[0 for _ in range(num_questions + 2)] for _ in range(num_interviews)]

    # iterate over files in that directory to populate the user response matrix
    current_interview = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and "DS_Store" not in f:
            # open each file as a csv
            # code to read csvs: https://remarkablemark.org/blog/2020/08/26/python-iterate-csv-rows/
            with open(f, 'r') as csvfile:
                datareader = csv.reader(csvfile)
                next(datareader)  # skip first row
                current_question_key = 0 # start by documenting the intro chit chat
                for row in datareader:
                    if len(row) > 0: # Accounts for empty rows in the csv
                        row_string = row[0]
                        [start_time, end_time, speaker, comment] = row_string.split('\t')

                        # See if what Ellie said is a relevant question
                        if speaker == "Ellie":
                            if comment in question_map_dict.keys():
                                master_question = question_map_dict[comment]
                                current_question_key = question_key_dict[master_question]
                        elif speaker == "Participant":
                            current_response = interview_responses[current_interview][current_question_key]
                            # If a response hasn't been added yet, replace it with the comment's text
                            if current_response == 0:
                                interview_responses[current_interview][current_question_key] = comment
                            # If there is already a response. Append the new comment to the existing response
                            else:
                                interview_responses[current_interview][current_question_key] += (" " + comment)

            # Add the participant ID to the last column of the interview's row
            participant_ID = int(filename.split("_")[0])
            interview_responses[current_interview][num_questions + 1] = participant_ID

            #Increase the interview count when you move on to the next file
            current_interview += 1

    # ### Make an Excel with the response matrix
    # workbook = xlsxwriter.Workbook("Full_Response_Matrix.xlsx")
    # response_wkst = workbook.add_worksheet("Response Matrix")
    #
    # # Make the first two rows (question ID and question):
    # response_wkst.write(0, 0, "Interview ID")
    # response_wkst.write(1, 0, "Interview ID")
    # response_wkst.write(0, 1, 0)
    # response_wkst.write(1, 1, "Intro")
    # for i in range(2, num_questions + 2):
    #     response_wkst.write(0, i, i-1)
    #     master_question = question_key_dict_reversed[i-1]
    #     response_wkst.write(1, i, master_question)
    #
    # # Add responses for each interview
    # for row in range(len(interview_responses)):
    #     for col in range(len(interview_responses[0])):
    #         # Write the Interview ID first
    #         if col == len(interview_responses[0]) - 1:
    #             response_wkst.write(row + 2, 0, interview_responses[row][col])
    #         else:
    #             response_wkst.write(row + 2, col + 1, interview_responses[row][col])
    # workbook.close()
    # ### END: Make an Excel

    return (interview_responses, question_key_dict_reversed)

def clean_responses(interview_responses):
    cleaned_responses = copy.deepcopy(interview_responses)
    straight_responses = copy.deepcopy(interview_responses)
    # cleaned_responses = interview_responses.copy()
    num_rows = len(cleaned_responses)
    num_cols = len(cleaned_responses[0])

    stop_words = set(stopwords.words('english'))

    count = 0
    for r in range(num_rows):
        for c in range(num_cols - 1): # Leave out the last column with the interview ID
            response = interview_responses[r][c]
            if response != 0:
                ## Remove anything in <> brackets (e.g. <laugh>)
                removed_brackets = re.sub(r'<[^>]*>', '', response)
                ## Lowercase converstion is not necessary, because all the words are already lowercase
                ## Then tokenize the responses
                tokens = word_tokenize(removed_brackets)
                # Remove stop words
                filtered_words = [w for w in tokens if w not in stop_words]
                # Stem the words
                stemmed_words = [stemmer.stem(word) for word in filtered_words]
                cleaned_responses[r][c] = stemmed_words
                straight_responses[r][c] = removed_brackets
            else:
                cleaned_responses[r][c] = ""
                straight_responses[r][c] = ""

    # # Make an Excel with the Cleaned response matrix
    # workbook = xlsxwriter.Workbook("Cleaned_Response_Matrix.xlsx")
    # response_wkst = workbook.add_worksheet("Clean Matrix")
    #
    # # Make the first two rows (question ID and question):
    # response_wkst.write(0, 0, "Interview ID")
    # response_wkst.write(1, 0, "Interview ID")
    # response_wkst.write(0, 1, 0)
    # response_wkst.write(1, 1, "Intro")
    # num_questions = len(question_key_dict_reversed.keys())
    # for i in range(2, num_questions + 2):
    #     response_wkst.write(0, i, i-1)
    #     master_question = question_key_dict_reversed[i-1]
    #     response_wkst.write(1, i, master_question)
    #
    # # Add responses for each interview
    # for row in range(len(cleaned_responses)):
    #     for col in range(len(cleaned_responses[0])):
    #         # Write the Interview ID first
    #         if col == len(cleaned_responses[0]) - 1:
    #             response_wkst.write(row + 2, 0, str(cleaned_responses[row][col]))
    #         else:
    #             response_wkst.write(row + 2, col + 1, str(cleaned_responses[row][col]))
    # workbook.close()
    return (cleaned_responses, straight_responses)

# Return a response matrix that just focuses on the "selected questions" (the questions we want to train the model on)
# The response matrix will be a numpy array
def selected_questions(cleaned_responses):
    # Read in the CSV file that indicates which questions will be selected
    responses_w_IDs = [] # List of lists that have just the columns corresponding to the selected questions
    no_IDs_responses = [] # Same as selected responses but without the interview IDs
    combined_responses = [] # Just has 1 column with the concatenated answers to all the questions
    selection_dict = {}  # dictionary with question index/key as key and a binary variable (selected = 1, not selected = 0) as the value
    selection_csv = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/Selected-EMMA-Questions.csv"
    with open(selection_csv, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)  # skip first row
        for row in datareader:
            if len(row) > 0:  # Accounts for empty rows in the csv
                [master_question, index, frequency, use_in_matrix] = row
                selection_dict[int(index)] = int(use_in_matrix)

    # Iterate through the cleaned responses to only pull out the columns that correspond to the selected questions
    num_rows = len(cleaned_responses)
    num_cols = len(cleaned_responses[0])
    questions_kept = [] # list of all the columns kept, in order
    for r in range(num_rows):
        row_list = []
        combined_row = []
        for c in range(num_cols - 1):  # Leave out the last column with the interview ID
            c_selected_check = selection_dict[c]
            if c_selected_check == 1:
                row_list.append(cleaned_responses[r][c])
                combined_row += cleaned_responses[r][c]
                if c not in questions_kept:
                    questions_kept.append(c)
        no_IDs_responses.append(row_list) # does not include the interview ID
        row_list.insert(0, cleaned_responses[r][num_cols-1]) # Add the interview ID to the start of the matrix
        responses_w_IDs.append(row_list)
        combined_responses.append(combined_row)

    # # Make an Excel with the selected response matrix
    # workbook = xlsxwriter.Workbook("Selected_Response_Matrix.xlsx")
    # response_wkst = workbook.add_worksheet("Selected Matrix")
    # # Add responses for each interview
    # for row in range(len(responses_w_IDs)):
    #     for col in range(len(responses_w_IDs[0])):
    #         response_wkst.write(row, col, str(responses_w_IDs[row][col]))
    # workbook.close()

    return [responses_w_IDs, combined_responses, questions_kept]

# Return a label matrix in the form of a numpy array
# Number of rows: number of interviews; one column with the label: depressed - 1, not depressed - 0
def label_array(selected_responses):
    # Read in the CSV file with the labels
    label_list = [] # List of lists that have one row for each interview and one element per row (the label)
    label_dict = {} # dictionary where the interview ID is the key and the depression label is the value
    label_csv = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/Detailed_PHQ8_Labels.csv"
    # Make the label dictionary
    with open(label_csv, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)  # skip first row
        for row in datareader:
            if len(row) > 0:  # Accounts for empty rows in the csv
                [ID, val1, val2, val3, val4, val5, val6, val7, val8, val9, depressed_flag] = row
                label_dict[int(ID)] = int(depressed_flag)

    # Iterate row by row through the response matrix. Match up the label matrix to the rows of the response matrix
    num_rows = len(selected_responses)
    for r in range(num_rows):
        interviewID = int(selected_responses[r][0])
        label = label_dict[interviewID]
        label_list.append([label])

    return label_list