from utils_gui import *
from model_diagnosis import *
from speech_to_text import *
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
    image_dashboard_results, image_file_dashboard_results, \
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


    image_file_dashboard_results = PhotoImage(
        file=relative_to_assets("image_dashboard_results.png"))

    image_dashboard_results = Label(
        frame_results,
        image=image_file_dashboard_results,
        bd=0
    )

    image_dashboard_results.place(
        x=52,
        y=133,
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


    diagnosis, confidence = model_diagnosis()

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
        y=389.0,
        width=498.0,
        height=60.0
    )

    data_confidence_results = Label(
        frame_results,
        text= confidence,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        font =('DM Sans',30,'bold'),
        anchor="w", justify='left'
    )
    data_confidence_results.place(
        x=324.0,
        y=462.0,
        width=75.0,
        height=60.0
    )
    

    return frame_results


def button_download_transcript_button():
    df = pd.DataFrame(speech_to_text())
    df.to_csv('/Users/reginaceballos/Downloads/EMMA_interview.csv', index=False, header=False)

