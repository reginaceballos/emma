from utils_gui import *
from tkinter import Canvas, PhotoImage, Label, Button, Frame


def button_repeat_hover_q5(e):
    button_repeat_q5.config(
        image=button_repeat_image_hover_q5
    )
def button_repeat_leave_q5(e):
    button_repeat_q5.config(
        image=button_repeat_image_q5
    )

def button_record_hover_q5(e):
    button_record_q5.config(
        image=button_record_image_hover_q5
    )
def button_record_leave_q5(e):
    button_record_q5.config(
        image=button_record_image_q5
    )

def button_play_hover_q5(e):
    button_play_q5.config(
        image=button_play_image_hover_q5
    )
def button_play_leave_q5(e):
    button_play_q5.config(
        image=button_play_image_q5
    )

def button_next_hover_q5(e):
    button_next_q5.config(
        image=button_next_image_hover_q5
    )
def button_next_leave_q5(e):
    button_next_q5.config(
        image=button_next_image_q5
    )

def create_gui_q5(window, next_frame, next_first_ask_function):
    global \
    image_background_q5, image_file_background_q5, \
    image_header_q5, image_file_header_q5, \
    image_question_q5, image_file_question_q5, \
    button_repeat_q5, button_repeat_image_q5, button_repeat_image_hover_q5, \
    button_record_q5, button_record_image_q5, button_record_image_hover_q5, \
    button_play_q5, button_play_image_q5, button_play_image_hover_q5, \
    button_next_q5, button_next_image_q5, button_next_image_hover_q5

    frame_q5 = Frame(window,
                     height=800,
                     width=1400)
    frame_q5.place(x=0, y=0)


    canvas_q5 = Canvas(
        frame_q5,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_q5.place(x=0, y=0)

    image_file_background_q5 = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_q5 = Label(
        frame_q5,
        image=image_file_background_q5,
        bd=0
    )
    image_background_q5.place(
        x=0,
        y=0,
    )

    image_file_header_q5 = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_q5 = Label(
        frame_q5,
        image=image_file_header_q5,
        bd=0
    )
    image_header_q5.place(
        x=0,
        y=0,
    )


    image_file_question_q5 = PhotoImage(
        file=relative_to_assets("image_question_q5.png"))

    image_question_q5 = Label(
        frame_q5,
        image=image_file_question_q5,
        bd=0
    )

    image_question_q5.place(
        x=0,
        y=167,
    )

    button_repeat_image_q5 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))
    button_repeat_q5 = Button(
        frame_q5,
        image=button_repeat_image_q5,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: play_audio('ask_questions_speech/ask_q5.wav'),
        relief="flat",
        state="disabled"
    )
    button_repeat_q5.place(
        x=175.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_repeat_image_hover_q5 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))

    button_repeat_q5.bind('<Enter>', button_repeat_hover_q5)
    button_repeat_q5.bind('<Leave>', button_repeat_leave_q5)

    button_record_image_q5 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))
    button_record_q5 = Button(
        frame_q5,
        image=button_record_image_q5,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: recording_button('q5'),
        relief="flat"
    )
    button_record_q5.place(
        x=486.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_record_image_hover_q5 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))

    button_record_q5.bind('<Enter>', button_record_hover_q5)
    button_record_q5.bind('<Leave>', button_record_leave_q5)

    button_play_image_q5 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))
    button_play_q5 = Button(
        frame_q5,
        image=button_play_image_q5,
        highlightthickness=0,
        command=lambda: play_audio("answer_questions_speech/recording_q5.wav"),
        relief="flat",
        state="disabled"
    )
    button_play_q5.place(
        x=797.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_play_image_hover_q5 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))

    button_play_q5.bind('<Enter>', button_play_hover_q5)
    button_play_q5.bind('<Leave>', button_play_leave_q5)

    button_next_image_q5 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))
    button_next_q5 = Button(
        frame_q5,
        image=button_next_image_q5,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: move_to_next_question_frame(next_frame, next_first_ask_function),
        relief="flat",
        state="disabled"
    )
    button_next_q5.place(
        x=1108.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_next_image_hover_q5 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))

    button_next_q5.bind('<Enter>', button_next_hover_q5)
    button_next_q5.bind('<Leave>', button_next_leave_q5)

    return frame_q5

record = False


def recording_button(file_suffix):
    global record, \
        button_repeat_q5, button_repeat_image_q5, button_repeat_image_hover_q5,\
        button_record_q5, button_record_image_q5, button_record_image_hover_q5, \
        button_play_q5, button_play_image_q5, button_play_image_hover_q5, \
        button_next_q5, button_next_image_q5, button_next_image_hover_q5

    if record:
        stop()

        button_record_image_q5 = PhotoImage(file=relative_to_assets("button_record.png"))
        button_record_image_hover_q5 = PhotoImage(file=relative_to_assets("button_record_hover.png"))
        button_record_q5["image"] = button_record_image_hover_q5

        button_repeat_image_q5 = PhotoImage(file=relative_to_assets("button_repeat.png"))
        button_repeat_image_hover_q5 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
        button_repeat_q5["image"] = button_repeat_image_q5
        button_repeat_q5["state"] = "normal"

        button_play_image_q5 = PhotoImage(file=relative_to_assets("button_play.png"))
        button_play_image_hover_q5 = PhotoImage(file=relative_to_assets("button_play_hover.png"))
        button_play_q5["image"] = button_play_image_q5
        button_play_q5["state"] = "normal"

        button_next_image_q5 = PhotoImage(file=relative_to_assets("button_next.png"))
        button_next_image_hover_q5 = PhotoImage(file=relative_to_assets("button_next_hover.png"))
        button_next_q5["image"] = button_next_image_q5
        button_next_q5["state"] = "normal"

    else:
        start(file_suffix)

        button_record_image_q5 = PhotoImage(file=relative_to_assets("button_stop_recording.png"))
        button_record_image_hover_q5 = PhotoImage(file=relative_to_assets("button_stop_recording_hover.png"))
        button_record_q5["image"] = button_record_image_hover_q5

        button_repeat_image_q5 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_image_hover_q5 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_q5["image"] = button_repeat_image_q5
        button_repeat_q5["state"] = "disabled"

        button_play_image_q5 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_image_hover_q5 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_q5["image"] = button_play_image_q5
        button_play_q5["state"] = "disabled"

        button_next_image_q5 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_image_hover_q5 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_q5["image"] = button_next_image_q5
        button_next_q5["state"] = "disabled"

    record = not record



def first_ask_q5():
    global \
        button_repeat_q5, button_repeat_image_q5, button_repeat_image_hover_q5, \
        button_record_q5, button_record_image_q5, button_record_image_hover_q5

    play_audio('ask_questions_speech/ask_q5.wav')

    button_repeat_image_q5 = PhotoImage(file=relative_to_assets("button_repeat.png"))
    button_record_image_q5 = PhotoImage(file=relative_to_assets("button_record.png"))

    button_repeat_image_hover_q5 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
    button_record_image_hover_q5 = PhotoImage(file=relative_to_assets("button_record_hover.png"))

    button_repeat_q5["image"] = button_repeat_image_q5
    button_record_q5["image"] = button_record_image_q5

    button_repeat_q5["state"] = "normal"
    button_record_q5["state"] = "normal"
