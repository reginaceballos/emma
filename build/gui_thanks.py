from utils_gui import *
from tkinter import Canvas, PhotoImage, Label, Button, Frame



def create_gui_thanks(window):
    global \
    image_background_thanks, image_file_background_thanks, \
    image_header_thanks, image_file_header_thanks, \
    image_question_thanks, image_file_question_thanks

    frame_thanks = Frame(window,
                     height=800,
                     width=1400)
    frame_thanks.place(x=0, y=0)


    canvas_thanks = Canvas(
        frame_thanks,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_thanks.place(x=0, y=0)

    image_file_background_thanks = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_thanks = Label(
        frame_thanks,
        image=image_file_background_thanks,
        bd=0
    )
    image_background_thanks.place(
        x=0,
        y=0,
    )

    image_file_header_thanks = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_thanks = Label(
        frame_thanks,
        image=image_file_header_thanks,
        bd=0
    )
    image_header_thanks.place(
        x=0,
        y=0,
    )


    image_file_question_thanks = PhotoImage(
        file=relative_to_assets("image_question_thanks.png"))

    image_question_thanks = Label(
        frame_thanks,
        image=image_file_question_thanks,
        bd=0
    )

    image_question_thanks.place(
        x=0,
        y=191,
    )

    return frame_thanks




def first_ask_thanks():
    play_audio('ask_questions_speech/ask_thanks.wav')
