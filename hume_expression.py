from hume import HumeBatchClient
from hume.models.config import FaceConfig, ProsodyConfig
import pandas as pd
import numpy as np

def weighted_average(df):
    return (df['duration'] * df['score']).sum() / df['duration'].sum()


def facial_expression_scores():
    client = HumeBatchClient("YNGcwNgXieBYk18On8c9WW2e2dljupo1ZsYF34ZbTaK5sXDd", timeout=None)
    filepaths = [
        "answer_questions_video/video_answer_q1.mp4",
        "answer_questions_video/video_answer_q4.mp4",
        "answer_questions_video/video_answer_q5.mp4",
        "answer_questions_video/video_answer_q6.mp4",
        "answer_questions_video/video_answer_q30.mp4",
    ]
    config = FaceConfig()
    job = client.submit_job(None, [config], files=filepaths)
    print(job)
    print("Running...")
    details = job.await_complete()


    json = job.get_predictions()


    df = []


    for f in json:
        for file_data in f['results']['predictions']:
            for predictions in file_data['models']['face']['grouped_predictions']:
                for prediction_frame in predictions['predictions']:
                    for emotion in prediction_frame['emotions']:
                        print(emotion)
                        l = []
                        l.append(f['source']['filename'])
                        l.append(prediction_frame['time'])
                        l.append(emotion['name'])
                        l.append(emotion['score'])
                        df.append(l)



    df = pd.DataFrame(df, columns=['file', 'time', 'emotion', 'score'])

    df = df.sort_values(['file','emotion','time']).reset_index().drop(columns='index')


    df['duration'] = df.groupby(['file','emotion'])['time'].rolling(2).apply(lambda x: x.max() - x.min()).reset_index(level=[0,1], drop=True)
    df['duration'][df['duration'].isna()] = 0

    facial_sadness, facial_disgust, facial_joy = df.groupby('emotion').apply(weighted_average)[['Sadness', 'Disgust', 'Joy']]

    return facial_sadness, facial_disgust, facial_joy




def speech_expression_scores():
    client = HumeBatchClient("YNGcwNgXieBYk18On8c9WW2e2dljupo1ZsYF34ZbTaK5sXDd", timeout=None)
    filepaths = [
    "answer_questions_speech/recording_q1.wav",
    "answer_questions_speech/recording_q4.wav",
    "answer_questions_speech/recording_q5.wav",
    "answer_questions_speech/recording_q6.wav",
    "answer_questions_speech/recording_q30.wav",
    ]
    config = ProsodyConfig()
    job = client.submit_job(None, [config], files=filepaths)
    print(job)
    print("Running...")
    details = job.await_complete()


    json = job.get_predictions()


    df = []

    for f in json:
        for file_data in f['results']['predictions']:
            for predictions in file_data['models']['prosody']['grouped_predictions']:
                for prediction_frame in predictions['predictions']:
                    for emotion in prediction_frame['emotions']:
                        l = []
                        l.append(f['source']['filename'])
                        l.append(prediction_frame['time']['begin'])
                        l.append(prediction_frame['time']['end'])
                        l.append(emotion['name'])
                        l.append(emotion['score'])
                        df.append(l)



    df = pd.DataFrame(df, columns=['file', 'start', 'end', 'emotion', 'score'])

    df['duration'] = df['end'] - df['start']

    speech_sadness, speech_disgust, speech_joy = df.groupby('emotion').apply(weighted_average)[['Sadness', 'Disgust', 'Joy']]

    return speech_sadness, speech_disgust, speech_joy

