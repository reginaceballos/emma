from utils_gui import *
from tkinter import Canvas, PhotoImage, Label, Button, Frame


def button_repeat_hover_q4(e):
    button_repeat_q4.config(
        image=button_repeat_image_hover_q4
    )
def button_repeat_leave_q4(e):
    button_repeat_q4.config(
        image=button_repeat_image_q4
    )

def button_record_hover_q4(e):
    button_record_q4.config(
        image=button_record_image_hover_q4
    )
def button_record_leave_q4(e):
    button_record_q4.config(
        image=button_record_image_q4
    )

def button_play_hover_q4(e):
    button_play_q4.config(
        image=button_play_image_hover_q4
    )
def button_play_leave_q4(e):
    button_play_q4.config(
        image=button_play_image_q4
    )

def button_next_hover_q4(e):
    button_next_q4.config(
        image=button_next_image_hover_q4
    )
def button_next_leave_q4(e):
    button_next_q4.config(
        image=button_next_image_q4
    )

def create_gui_q4(window, next_frame, next_first_ask_function):
    global \
    image_background_q4, image_file_background_q4, \
    image_header_q4, image_file_header_q4, \
    image_question_q4, image_file_question_q4, \
    button_repeat_q4, button_repeat_image_q4, button_repeat_image_hover_q4, \
    button_record_q4, button_record_image_q4, button_record_image_hover_q4, \
    button_play_q4, button_play_image_q4, button_play_image_hover_q4, \
    button_next_q4, button_next_image_q4, button_next_image_hover_q4

    frame_q4 = Frame(window,
                     height=800,
                     width=1400)
    frame_q4.place(x=0, y=0)


    canvas_q4 = Canvas(
        frame_q4,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_q4.place(x=0, y=0)

    image_file_background_q4 = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_q4 = Label(
        frame_q4,
        image=image_file_background_q4,
        bd=0
    )
    image_background_q4.place(
        x=0,
        y=0,
    )

    image_file_header_q4 = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_q4 = Label(
        frame_q4,
        image=image_file_header_q4,
        bd=0
    )
    image_header_q4.place(
        x=0,
        y=0,
    )


    image_file_question_q4 = PhotoImage(
        file=relative_to_assets("image_question_q4.png"))

    image_question_q4 = Label(
        frame_q4,
        image=image_file_question_q4,
        bd=0
    )

    image_question_q4.place(
        x=0,
        y=210,
    )

    button_repeat_image_q4 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))
    button_repeat_q4 = Button(
        frame_q4,
        image=button_repeat_image_q4,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: play_audio('ask_questions_speech/ask_q4.wav'),
        relief="flat",
        state="disabled"
    )
    button_repeat_q4.place(
        x=175.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_repeat_image_hover_q4 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))

    button_repeat_q4.bind('<Enter>', button_repeat_hover_q4)
    button_repeat_q4.bind('<Leave>', button_repeat_leave_q4)

    button_record_image_q4 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))
    button_record_q4 = Button(
        frame_q4,
        image=button_record_image_q4,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: recording_button('q4'),
        relief="flat"
    )
    button_record_q4.place(
        x=486.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_record_image_hover_q4 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))

    button_record_q4.bind('<Enter>', button_record_hover_q4)
    button_record_q4.bind('<Leave>', button_record_leave_q4)

    button_play_image_q4 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))
    button_play_q4 = Button(
        frame_q4,
        image=button_play_image_q4,
        highlightthickness=0,
        command=lambda: play_audio("answer_questions_speech/recording_q4.wav"),
        relief="flat",
        state="disabled"
    )
    button_play_q4.place(
        x=797.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_play_image_hover_q4 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))

    button_play_q4.bind('<Enter>', button_play_hover_q4)
    button_play_q4.bind('<Leave>', button_play_leave_q4)

    button_next_image_q4 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))
    button_next_q4 = Button(
        frame_q4,
        image=button_next_image_q4,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: move_to_next_question_frame(next_frame, next_first_ask_function),
        relief="flat",
        state="disabled"
    )
    button_next_q4.place(
        x=1108.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_next_image_hover_q4 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))

    button_next_q4.bind('<Enter>', button_next_hover_q4)
    button_next_q4.bind('<Leave>', button_next_leave_q4)

    return frame_q4

record = False


def recording_button(file_suffix):
    global record, \
        button_repeat_q4, button_repeat_image_q4, button_repeat_image_hover_q4,\
        button_record_q4, button_record_image_q4, button_record_image_hover_q4, \
        button_play_q4, button_play_image_q4, button_play_image_hover_q4, \
        button_next_q4, button_next_image_q4, button_next_image_hover_q4

    if record:
        stop()

        button_record_image_q4 = PhotoImage(file=relative_to_assets("button_record.png"))
        button_record_image_hover_q4 = PhotoImage(file=relative_to_assets("button_record_hover.png"))
        button_record_q4["image"] = button_record_image_hover_q4

        button_repeat_image_q4 = PhotoImage(file=relative_to_assets("button_repeat.png"))
        button_repeat_image_hover_q4 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
        button_repeat_q4["image"] = button_repeat_image_q4
        button_repeat_q4["state"] = "normal"

        button_play_image_q4 = PhotoImage(file=relative_to_assets("button_play.png"))
        button_play_image_hover_q4 = PhotoImage(file=relative_to_assets("button_play_hover.png"))
        button_play_q4["image"] = button_play_image_q4
        button_play_q4["state"] = "normal"

        button_next_image_q4 = PhotoImage(file=relative_to_assets("button_next.png"))
        button_next_image_hover_q4 = PhotoImage(file=relative_to_assets("button_next_hover.png"))
        button_next_q4["image"] = button_next_image_q4
        button_next_q4["state"] = "normal"

    else:
        start(file_suffix)

        button_record_image_q4 = PhotoImage(file=relative_to_assets("button_stop_recording.png"))
        button_record_image_hover_q4 = PhotoImage(file=relative_to_assets("button_stop_recording_hover.png"))
        button_record_q4["image"] = button_record_image_hover_q4

        button_repeat_image_q4 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_image_hover_q4 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_q4["image"] = button_repeat_image_q4
        button_repeat_q4["state"] = "disabled"

        button_play_image_q4 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_image_hover_q4 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_q4["image"] = button_play_image_q4
        button_play_q4["state"] = "disabled"

        button_next_image_q4 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_image_hover_q4 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_q4["image"] = button_next_image_q4
        button_next_q4["state"] = "disabled"

    record = not record



def first_ask_q4():
    global \
        button_repeat_q4, button_repeat_image_q4, button_repeat_image_hover_q4, \
        button_record_q4, button_record_image_q4, button_record_image_hover_q4

    play_audio('ask_questions_speech/ask_q4.wav')

    button_repeat_image_q4 = PhotoImage(file=relative_to_assets("button_repeat.png"))
    button_record_image_q4 = PhotoImage(file=relative_to_assets("button_record.png"))

    button_repeat_image_hover_q4 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
    button_record_image_hover_q4 = PhotoImage(file=relative_to_assets("button_record_hover.png"))

    button_repeat_q4["image"] = button_repeat_image_q4
    button_record_q4["image"] = button_record_image_q4

    button_repeat_q4["state"] = "normal"
    button_record_q4["state"] = "normal"
