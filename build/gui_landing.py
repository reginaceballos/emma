from utils_gui import *
from tkinter import Canvas, PhotoImage, Label, Button, Frame


def button_start_hover(e):
    button_start.config(
        image=button_start_image_hover
    )
def button_start_leave(e):
    button_start.config(
        image=button_start_image
    )

def create_gui_landing(window, next_frame, next_first_ask_function):
    global \
    image_background_landing, image_file_background_landing, \
    image_header_landing, image_file_header_landing, \
    image_question_landing, image_file_question_landing, \
    button_start, button_start_image, button_start_image_hover

    frame_landing = Frame(window,
                     height=800,
                     width=1400)
    frame_landing.place(x=0, y=0)


    canvas_landing = Canvas(
        frame_landing,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_landing.place(x=0, y=0)

    image_file_background_landing = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_landing = Label(
        frame_landing,
        image=image_file_background_landing,
        bd=0
    )
    image_background_landing.place(
        x=0,
        y=0,
    )

    image_file_header_landing = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_landing = Label(
        frame_landing,
        image=image_file_header_landing,
        bd=0
    )
    image_header_landing.place(
        x=0,
        y=0,
    )


    image_file_question_landing = PhotoImage(
        file=relative_to_assets("image_question_landing.png"))

    image_question_landing = Label(
        frame_landing,
        image=image_file_question_landing,
        bd=0
    )

    image_question_landing.place(
        x=0,
        y=210,
    )

    button_start_image = PhotoImage(
        file=relative_to_assets("button_start_disabled.png"))
    button_start = Button(
        frame_landing,
        image=button_start_image,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: move_to_next_question_frame(next_frame, next_first_ask_function),
        relief="flat",
        state="disabled"
    )

    button_start.place(
        x=520.0,
        y=607.0,
        width=359.0,
        height=68.0
    )

    button_start_image_hover = PhotoImage(
        file=relative_to_assets("button_start_disabled.png"))

    button_start.bind('<Enter>', button_start_hover)
    button_start.bind('<Leave>', button_start_leave)

    return frame_landing




def first_ask_landing():
    global \
        button_start, button_start_image, button_start_image_hover

    play_audio('ask_questions_speech/ask_landing.wav')

    button_start_image = PhotoImage(file=relative_to_assets("button_start.png"))

    button_start_image_hover = PhotoImage(file=relative_to_assets("button_start_hover.png"))

    button_start["image"] = button_start_image

    button_start["state"] = "normal"
