from utils_gui import *
from naive_bayes.Make_New_Prediction import *
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
        x=756,
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

    if confidence.isnumeric():
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
        width=498.0,
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
        y=472.0,
        width=80.0,
        height=60.0
    )
    

    facial_sadness, facial_disgust, facial_joy = facial_expression_scores()
    # facial_sadness, facial_disgust, facial_joy = np.random.rand(), np.random.rand(), np.random.rand()

    data_facial_sadness = Label(
        frame_results,
        text= '{:.0%}'.format(facial_sadness),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_sadness.place(
        x=1016.0,
        y=400.0,
        width=80.0,
        height=60.0
    )


    data_facial_disgust = Label(
        frame_results,
        text= '{:.0%}'.format(facial_disgust),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_disgust.place(
        x=1016.0,
        y=472.0,
        width=80.0,
        height=60.0
    )


    data_facial_joy = Label(
        frame_results,
        text= '{:.0%}'.format(facial_joy),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_facial_joy.place(
        x=1016.0,
        y=582.0,
        width=80.0,
        height=60.0
    )


    speech_sadness, speech_disgust, speech_joy = speech_expression_scores()
    # speech_sadness, speech_disgust, speech_joy = np.random.rand(), np.random.rand(), np.random.rand()

    data_speech_sadness = Label(
        frame_results,
        text= '{:.0%}'.format(speech_sadness),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_sadness.place(
        x=1193.0,
        y=400.0,
        width=80.0,
        height=60.0
    )


    data_speech_disgust = Label(
        frame_results,
        text= '{:.0%}'.format(speech_disgust),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_disgust.place(
        x=1193.0,
        y=472.0,
        width=80.0,
        height=60.0
    )


    data_speech_joy = Label(
        frame_results,
        text= '{:.0%}'.format(speech_joy),
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_speech_joy.place(
        x=1193.0,
        y=582.0,
        width=80.0,
        height=60.0
    )



    return frame_results


def button_download_transcript_button():
    df = pd.DataFrame(speech_to_text())
    df.to_csv('/Users/reginaceballos/Downloads/EMMA_interview.csv', index=False, header=False)

