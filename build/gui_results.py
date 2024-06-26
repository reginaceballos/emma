from utils_gui import *
from Make_New_Prediction import *
from speech_to_text import *
from hume_expression import *
from tkinter import Canvas, PhotoImage, Label, Button, Frame


def button_download_transcript_hover(e):
    button_download_transcript.config(
        image=button_download_transcript_image_hover
    )
def button_download_transcript_leave(e):
    button_download_transcript.config(
        image=button_download_transcript_image
    )

def create_gui_results(window):
    global \
    image_background_results, image_file_background_results, \
    image_header_results, image_file_header_results, \
    image_dashboard_diagnosis_results, image_file_dashboard_diagnosis_results, \
    image_dashboard_emotion_results, image_file_dashboard_emotion_results, \
    button_download_transcript, button_download_transcript_image, button_download_transcript_image_hover

    frame_results = Frame(window,
                     height=800,
                     width=1400)
    frame_results.place(x=0, y=0)


    image_file_background_results = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_results = Label(
        frame_results,
        image=image_file_background_results,
        bd=0
    )
    image_background_results.place(
        x=0,
        y=0,
    )

    image_file_header_results = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_results = Label(
        frame_results,
        image=image_file_header_results,
        bd=0
    )
    image_header_results.place(
        x=0,
        y=0,
    )


    image_file_dashboard_diagnosis_results = PhotoImage(
        file=relative_to_assets("image_dashboard_diagnosis_results.png"))

    image_dashboard_diagnosis_results = Label(
        frame_results,
        image=image_file_dashboard_diagnosis_results,
        bd=0
    )

    image_dashboard_diagnosis_results.place(
        x=52,
        y=133,
    )
    

    image_file_dashboard_emotion_results = PhotoImage(
        file=relative_to_assets("image_dashboard_emotion_results.png"))

    image_dashboard_emotion_results = Label(
        frame_results,
        image=image_file_dashboard_emotion_results,
        bd=0
    )

    image_dashboard_emotion_results.place(
        x=734,
        y=221,
    )

    image_file_line_results = PhotoImage(
        file=relative_to_assets("image_line_results.png"))

    image_line_results = Label(
        frame_results,
        image=image_file_line_results,
        bd=0
    )

    image_line_results.place(
        x=700,
        y=221,
    )
    

    button_download_transcript_image = PhotoImage(
        file=relative_to_assets("button_download_transcript.png"))
    button_download_transcript = Button(
        frame_results,
        image=button_download_transcript_image,
        borderwidth=0,
        highlightthickness=0,
        command=button_download_transcript_button,
        relief="flat"
    )

    button_download_transcript.place(
        x=176.0,
        y=635.0,
        width=359.0,
        height=68.0
    )

    button_download_transcript_image_hover = PhotoImage(
        file=relative_to_assets("button_download_transcript_hover.png"))

    button_download_transcript.bind('<Enter>', button_download_transcript_hover)
    button_download_transcript.bind('<Leave>', button_download_transcript_leave)


    diagnosis, confidence = make_new_prediction()

    if isinstance(confidence, float) or confidence.isnumeric():
        confidence_str = '{:.0%}'.format(confidence)
    else:
        confidence_str = confidence

    data_diagnosis_results = Label(
        frame_results,
        text= diagnosis,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_diagnosis_results.place(
        x=107.0,
        y=400.0,
        width=505.0,
        height=60.0
    )

    data_confidence_results = Label(
        frame_results,
        text= confidence_str,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_confidence_results.place(
        x=324.0,
        y=475.0,
        width=80.0,
        height=60.0
    )
    

    # facial_sadness, facial_disgust, facial_joy = facial_expression_scores()
    # facial_sadness, facial_disgust, facial_joy = np.random.rand(), np.random.rand(), np.random.rand()

    df_top_facial_emotions = top_facial_emotions(3)

    facial_emotion_1 = df_top_facial_emotions['emotion'][0]
    facial_emotion_2 = df_top_facial_emotions['emotion'][1]
    facial_emotion_3 = df_top_facial_emotions['emotion'][2]

    # facial_emotion_1 = 'Sadness'
    # facial_emotion_2 = 'Tiredness'
    # facial_emotion_3 = 'Anxiety'

    facial_score_1 = df_top_facial_emotions['score'][0]
    facial_score_2 = df_top_facial_emotions['score'][1]
    facial_score_3 = df_top_facial_emotions['score'][2]

    # facial_score_1 = 0.47
    # facial_score_2 = 0.42
    # facial_score_3 = 0.39


    data_facial_emotion_1 = Label(
        frame_results,
        text= facial_emotion_1,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_emotion_1.place(
        x=965.0,
        y=323.0,
        width=250,
        height=60.0
    )

    data_facial_score_1 = Label(
        frame_results,
        text= '{:.0%}'.format(facial_score_1),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_score_1.place(
        x=1250.0,
        y=323.0,
        width=80.0,
        height=60.0
    )



    data_facial_emotion_2 = Label(
        frame_results,
        text= facial_emotion_2,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_emotion_2.place(
        x=965.0,
        y=371.0,
        width=250,
        height=60.0
    )

    data_facial_score_2 = Label(
        frame_results,
        text= '{:.0%}'.format(facial_score_2),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_score_2.place(
        x=1250.0,
        y=371.0,
        width=80.0,
        height=60.0
    )



    data_facial_emotion_3 = Label(
        frame_results,
        text= facial_emotion_3,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_emotion_3.place(
        x=965.0,
        y=419,
        width=250,
        height=60.0
    )

    data_facial_score_3 = Label(
        frame_results,
        text= '{:.0%}'.format(facial_score_3),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_score_3.place(
        x=1250.0,
        y=419,
        width=80.0,
        height=60.0
    )

    # data_facial_sadness = Label(
    #     frame_results,
    #     text= '{:.0%}'.format(facial_sadness),
    #     bd=0,
    #     bg="#7999D4",
    #     fg="#FFFFFF",
    #     highlightthickness=0,
    #     font =('DM Sans',30,'bold'),
    #     anchor="w", justify='left'
    # )
    # data_facial_sadness.place(
    #     x=1016.0,
    #     y=400.0,
    #     width=80.0,
    #     height=60.0
    # )


    # data_facial_disgust = Label(
    #     frame_results,
    #     text= '{:.0%}'.format(facial_disgust),
    #     bd=0,
    #     bg="#7999D4",
    #     fg="#FFFFFF",
    #     highlightthickness=0,
    #     font =('DM Sans',30,'bold'),
    #     anchor="w", justify='left'
    # )
    # data_facial_disgust.place(
    #     x=1016.0,
    #     y=472.0,
    #     width=80.0,
    #     height=60.0
    # )


    # data_facial_joy = Label(
    #     frame_results,
    #     text= '{:.0%}'.format(facial_joy),
    #     bd=0,
    #     bg="#7999D4",
    #     fg="#FFFFFF",
    #     highlightthickness=0,
    #     font =('DM Sans',30,'bold'),
    #     anchor="w", justify='left'
    # )
    # data_facial_joy.place(
    #     x=1016.0,
    #     y=582.0,
    #     width=80.0,
    #     height=60.0
    # )


    # speech_sadness, speech_disgust, speech_joy = speech_expression_scores()
    # speech_sadness, speech_disgust, speech_joy = np.random.rand(), np.random.rand(), np.random.rand()

    df_top_speech_emotions = top_speech_emotions(3)

    speech_emotion_1 = df_top_speech_emotions['emotion'][0]
    speech_emotion_2 = df_top_speech_emotions['emotion'][1]
    speech_emotion_3 = df_top_speech_emotions['emotion'][2]
    
    # speech_emotion_1 = 'Tiredness'
    # speech_emotion_2 = 'Sadness'
    # speech_emotion_3 = 'Boredom'

    speech_score_1 = df_top_speech_emotions['score'][0]
    speech_score_2 = df_top_speech_emotions['score'][1]
    speech_score_3 = df_top_speech_emotions['score'][2]

    # speech_score_1 = 0.16
    # speech_score_2 = 0.12
    # speech_score_3 = 0.11


    data_speech_emotion_1 = Label(
        frame_results,
        text= speech_emotion_1,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_emotion_1.place(
        x=965.0,
        y=547,
        width=250,
        height=60.0
    )

    data_speech_score_1 = Label(
        frame_results,
        text= '{:.0%}'.format(speech_score_1),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_score_1.place(
        x=1250.0,
        y=547,
        width=80.0,
        height=60.0
    )



    data_speech_emotion_2 = Label(
        frame_results,
        text= speech_emotion_2,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_emotion_2.place(
        x=965.0,
        y=595,
        width=250,
        height=60.0
    )

    data_speech_score_2 = Label(
        frame_results,
        text= '{:.0%}'.format(speech_score_2),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_score_2.place(
        x=1250.0,
        y=595,
        width=80.0,
        height=60.0
    )



    data_speech_emotion_3 = Label(
        frame_results,
        text= speech_emotion_3,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_emotion_3.place(
        x=965.0,
        y=643,
        width=250,
        height=60.0
    )

    data_speech_score_3 = Label(
        frame_results,
        text= '{:.0%}'.format(speech_score_3),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_score_3.place(
        x=1250.0,
        y=643,
        width=80.0,
        height=60.0
    )



    # data_speech_sadness = Label(
    #     frame_results,
    #     text= '{:.0%}'.format(speech_sadness),
    #     bd=0,
    #     bg="#7999D4",
    #     fg="#FFFFFF",
    #     highlightthickness=0,
    #     font =('DM Sans',30,'bold'),
    #     anchor="w", justify='left'
    # )
    # data_speech_sadness.place(
    #     x=1193.0,
    #     y=400.0,
    #     width=80.0,
    #     height=60.0
    # )


    # data_speech_disgust = Label(
    #     frame_results,
    #     text= '{:.0%}'.format(speech_disgust),
    #     bd=0,
    #     bg="#7999D4",
    #     fg="#FFFFFF",
    #     highlightthickness=0,
    #     font =('DM Sans',30,'bold'),
    #     anchor="w", justify='left'
    # )
    # data_speech_disgust.place(
    #     x=1193.0,
    #     y=472.0,
    #     width=80.0,
    #     height=60.0
    # )


    # data_speech_joy = Label(
    #     frame_results,
    #     text= '{:.0%}'.format(speech_joy),
    #     bd=0,
    #     bg="#7999D4",
    #     fg="#FFFFFF",
    #     highlightthickness=0,
    #     font =('DM Sans',30,'bold'),
    #     anchor="w", justify='left'
    # )
    # data_speech_joy.place(
    #     x=1193.0,
    #     y=582.0,
    #     width=80.0,
    #     height=60.0
    # )



    return frame_results


def button_download_transcript_button():
    df = pd.DataFrame(speech_to_text())
    # df.to_csv('/Users/reginaceballos/Downloads/EMMA_interview.csv', index=False, header=False)
    df.to_csv('EMMA_interview.csv', index=False, header=False)

