Summary of Key Items in the Repository:
1. main_interview_gui.py: Code to create the interface for the user interview. Uses the files in the build folder to create each individual frame of the interview: Welcome, Question 1, Question 4, Question 5, Question 6, Question 30, and Thank you frames.
1. main_results_gui.py: Code to create the interface for the results dashboard. Uses the files in the build folder to create each individual frame of the results: Login and Results frames.
1. Make_New_Prediction.py: Uses a pre-trained Naive Bayes model (naive_bayes_model.pkl) to predict if the user has depression; inputs: the text transcripts of the user's interview (e.g. Demo_Transcript.csv), vectorizers.pkl (the Bag of Words dictionaries for the Naive Bayes model), and saved_questions.pkl (a list of the questions asked in the interview). Output: a binary prediction of if the user has depression or not.
1. hume_expression.py: Calls the HumeAI API to calculate and display the emotions with the top scores on the results dashboard.
1. audio_recorder.py: Code with auxiliary functions to record and play audio files.
1. data_preparation.py: Uses natural language processing to convert the raw text output of interviews into cleaned tokens; tokenizes, stems, and removes stop words.
1. speech_to_text.py: Creates a dataframe with the two columns, the question key and the transcript of the users responses for each of the five questions asked.
1. utils_gui.py: Auxiliary functions to build the interfaces, including the configuration for the buttons play sounds and move to the next frame. 
1. Finetuned_Bert.py: Code to create and fine-tune a pre-trained BERT model (not used in the final EMMA implementation, but results are referenced in the writeup)
1. answer_questions_speech folder: Folder containing the video files in .mp4 format of the user recording themselves to answer to the questionnaire.
1. answer_questions_video folder: Folder containing the audio files in .wav format of the user recording themselves to answer to the questionnaire.
1. ask_questions_speech folder: Folder containing the audio files in .wav format of the EMMA asking each question.
1. build folder: Folder with the elements required to build the interfaces, including the individual interface for each frame and the images that make up each frame.
1. make_naive_bayes_model folder: folder containing the code and input data necessary to create a new Naive Bayes model; transcripts folder holds all the interviews we trained on; Detailed_PHQ8_Labels shows the label for each interview; Selected-EMMA-Questions lets you pick which questions the model is built from; Naive_Bayes_Separate_Qs.py trains a new Naive Bayes model and stores it as a pkl file for use in Make_New_Prediction.py

Steps to set up and run the EMMA system:
1. Install all the necessary packages: nltk (requires downloading stopwords from ssl), pickle, SpeechRecognition, tkinter, scikit-learn, Transformers, OpenCV, Pillow, tkvideo, pyaudio (requires installing portaudio using homebrew in Mac computers), wave, speechrecognition (requires installing flac using homebrew in Mac computers), and hume.
1. Run main_interview_gui.py and follow EMMA's instructions to record an interview
1. Run main_results_gui.py; enter the username "emma" and the password "1234" to access the results page
