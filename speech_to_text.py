import os
import speech_recognition as sr
import pandas as pd
import re


def speech_to_text(number_of_questions = 20):
    wd = os.path.abspath('')

    directory_questions_speech = os.path.abspath(wd + '/answer_questions_speech/')

    r = sr.Recognizer()

    text_answers = []

    i = 0

    for filename in os.listdir(directory_questions_speech):
        f = directory_questions_speech + '/' + filename

        # checking if it is a file
        if os.path.isfile(f) and filename.endswith('.wav'):
            question_audio = sr.AudioFile(f)

            with question_audio as source:
                audio = r.record(source)

                try:
                    s = r.recognize_google(audio)
                    text_answers.append([re.sub('\.wav', '', re.sub('.+recording_q', '', f)),
                                         s])

                except Exception as e:
                    print("Exception: " + str(e))

    df = pd.DataFrame(text_answers)

    return df


#print(speech_to_text())


# df.to_csv('EMMA_interview.csv', index=False, header=False)

