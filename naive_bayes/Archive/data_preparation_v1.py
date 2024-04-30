### Overview ###
# Make a matrix where the rows correspond to different users
# And the columns correspond to the questions in EMMA-Question-List.csv

# import libraries
import csv
import os
import re
import xlsxwriter

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

def response_matrix():
    # Determine how many unqiue "Master" questions there are
    question_key_dict = {} # dictionary with master question as key and the index (column number) as the value
    question_map_dict = {} # dictionary with sub question as key and master question as the value
    question_key_dict_reversed = {} # dictionary with the index as the key and the master question as the value
    question_list_csv = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/EMMA-Question-List.csv"
    with open(question_list_csv, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader) #skip first row
        for row in datareader:
            if len(row) > 0:  # Accounts for empty rows in the csv
                # print(row)
                [master_question, index, sub_question, frequency, use_in_matrix] = row
                if master_question not in question_key_dict:
                    question_key_dict[master_question] = int(index)
                    question_key_dict_reversed[int(index)] = master_question

                question_map_dict[sub_question] = master_question

    # Print question_map_dict
    # for (sub_question, master_question) in question_map_dict.items():
    #     print(sub_question, ", ", master_question)

    # # Print question_key_dict
    # for (master_question, key) in question_key_dict.items():
    #     print(master_question, ", ", key)

    num_questions = len(question_key_dict.keys())

    # Iterate through each interview and put their responses into matrix spot [user][question]
    directory = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/Transcripts"

    # Find the number of files in the folder
    num_interviews = 0
    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            num_interviews += 1
    # print('Number of Interviews:', num_interviews)

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
                # print(f)
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

    # Make an Excel with the response matrix
    workbook = xlsxwriter.Workbook("Full_Response_Matrix.xlsx")
    response_wkst = workbook.add_worksheet("Response Matrix")

    # Make the first two rows (question ID and question):
    response_wkst.write(0, 0, "Interview ID")
    response_wkst.write(1, 0, "Interview ID")
    response_wkst.write(0, 1, 0)
    response_wkst.write(1, 1, "Intro")
    for i in range(2, num_questions + 2):
        response_wkst.write(0, i, i-1)
        master_question = question_key_dict_reversed[i-1]
        response_wkst.write(1, i, master_question)

    # Add responses for each interview
    for row in range(len(interview_responses)):
        for col in range(len(interview_responses[0])):
            # Write the Interview ID first
            if col == len(interview_responses[0]) - 1:
                response_wkst.write(row + 2, 0, interview_responses[row][col])
            else:
                response_wkst.write(row + 2, col + 1, interview_responses[row][col])
    workbook.close()

    return interview_responses

def clean_responses(interview_responses):
    cleaned_responses = interview_responses.copy()
    num_rows = len(cleaned_responses)
    num_cols = len(cleaned_responses[0])
    count = 0
    for r in range(num_rows):
        for c in range(num_cols):
            print("INTERVIEW: ", interview_responses[r][num_cols-1])
            response = interview_responses[r][c]
            if response != 0:
                ## First remove anything in <> brackets (e.g. <laugh>)
                print("RESPONSE: ", response)
                removed_brackets = re.sub(r'<[^>]*>', '', response)
                print("REMOVED BRACKETS: ", removed_brackets)
                print("-------------------------------------------------------")

                # ## Then tokenize the responses
                # print("REMOVED BRACKETS: ", removed_brackets)
                # tokens = word_tokenize(removed_brackets)
                # print("TOKENS: ", tokens)
                # print("-------------------------------------------------------")
                count += 1
            if count > 0:
                break
        # if count > 0:
        #     break

    return 1
