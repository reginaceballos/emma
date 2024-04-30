# Code to read in files in a folder: https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/

# import required module
import os
import csv
import xlsxwriter

# assign directory
directory = "/Users/caeleyharihara/Documents/MIT/Spring 2024/Multimodal Interfaces/Final Project/EMMA/Transcripts"

count = 0
unique_comments_dict = {}
# iterate over files in that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
    # if os.path.isfile(f) and filename == '361_TRANSCRIPT.csv':
        print(filename)
        # open each file as a csv
        # code to read csvs: https://remarkablemark.org/blog/2020/08/26/python-iterate-csv-rows/
        with open(f, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                if len(row) > 0: # Accounts for empty rows in the csv
                    row_string = row[0]
                    row_list = row_string.split('\t')
                    speaker = row_list[2]

                    # Find all the unique things that Ellie said and count them
                    if speaker == "Ellie":
                        comment = row_list[3]
                        if comment in unique_comments_dict:
                            unique_comments_dict[comment] += 1
                        else:
                            unique_comments_dict[comment] = 1


        # count += 1
        # if count == 1:
        #     break

# print(unique_comments_dict)

## Make an Excel file with Ellie's unique comments
Ellie_comments_file_name = "Ellie-Unique-Statements.xlsx"
workbook = xlsxwriter.Workbook(Ellie_comments_file_name)

comments_wkst = workbook.add_worksheet("Unique Comments")
comments_wkst.write(0, 0, "Comment")
comments_wkst.write(0, 1, "Frequency")

row = 1
for (comment,freq) in unique_comments_dict.items():
    comments_wkst.write(row, 0, comment)
    comments_wkst.write(row, 1, freq)
    row += 1

workbook.close()