from utils_gui import *
from tkinter import Canvas, PhotoImage, Label, Button, Frame


def button_repeat_hover_q2(e):
    button_repeat_q2.config(
        image=button_repeat_image_hover_q2
    )
def button_repeat_leave_q2(e):
    button_repeat_q2.config(
        image=button_repeat_image_q2
    )

def button_record_hover_q2(e):
    button_record_q2.config(
        image=button_record_image_hover_q2
    )
def button_record_leave_q2(e):
    button_record_q2.config(
        image=button_record_image_q2
    )

def button_play_hover_q2(e):
    button_play_q2.config(
        image=button_play_image_hover_q2
    )
def button_play_leave_q2(e):
    button_play_q2.config(
        image=button_play_image_q2
    )

def button_next_hover_q2(e):
    button_next_q2.config(
        image=button_next_image_hover_q2
    )
def button_next_leave_q2(e):
    button_next_q2.config(
        image=button_next_image_q2
    )

def create_gui_q2(window):
    global \
    frame_q2, \
    image_background_q2, image_file_background_q2, \
    image_header_q2, image_file_header_q2, \
    image_question_q2, image_file_question_q2, \
    button_repeat_q2, button_repeat_image_q2, button_repeat_image_hover_q2, \
    button_record_q2, button_record_image_q2, button_record_image_hover_q2, \
    button_play_q2, button_play_image_q2, button_play_image_hover_q2, \
    button_next_q2, button_next_image_q2, button_next_image_hover_q2

    frame_q2 = Frame(window,
                     height=800,
                     width=1400)
    frame_q2.place(x=0, y=0)


    canvas_q2 = Canvas(
        frame_q2,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_q2.place(x=0, y=0)

    image_file_background_q2 = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_q2 = Label(
        frame_q2,
        image=image_file_background_q2,
        bd=0
    )
    image_background_q2.place(
        x=0,
        y=0,
    )

    image_file_header_q2 = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_q2 = Label(
        frame_q2,
        image=image_file_header_q2,
        bd=0
    )
    image_header_q2.place(
        x=0,
        y=0,
    )


    image_file_question_q2 = PhotoImage(
        file=relative_to_assets("image_question_q2.png"))

    image_question_q2 = Label(
        frame_q2,
        image=image_file_question_q2,
        bd=0
    )

    image_question_q2.place(
        x=0,
        y=210,
    )

    button_repeat_image_q2 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))
    button_repeat_q2 = Button(
        frame_q2,
        image=button_repeat_image_q2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: play_audio('ask_questions_speech/ask_q2.wav'),
        relief="flat",
        state="disabled"
    )
    button_repeat_q2.place(
        x=175.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_repeat_image_hover_q2 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))

    button_repeat_q2.bind('<Enter>', button_repeat_hover_q2)
    button_repeat_q2.bind('<Leave>', button_repeat_leave_q2)

    button_record_image_q2 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))
    button_record_q2 = Button(
        frame_q2,
        image=button_record_image_q2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: recording_button('q2'),
        relief="flat"
    )
    button_record_q2.place(
        x=486.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_record_image_hover_q2 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))

    button_record_q2.bind('<Enter>', button_record_hover_q2)
    button_record_q2.bind('<Leave>', button_record_leave_q2)

    button_play_image_q2 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))
    button_play_q2 = Button(
        frame_q2,
        image=button_play_image_q2,
        highlightthickness=0,
        command=lambda: play_audio("answer_questions_speech/recording_q2.wav"),
        relief="flat",
        state="disabled"
    )
    button_play_q2.place(
        x=797.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_play_image_hover_q2 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))

    button_play_q2.bind('<Enter>', button_play_hover_q2)
    button_play_q2.bind('<Leave>', button_play_leave_q2)

    button_next_image_q2 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))
    button_next_q2 = Button(
        frame_q2,
        image=button_next_image_q2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_4 clicked"),
        relief="flat",
        state="disabled"
    )
    button_next_q2.place(
        x=1108.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_next_image_hover_q2 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))

    button_next_q2.bind('<Enter>', button_next_hover_q2)
    button_next_q2.bind('<Leave>', button_next_leave_q2)

    return frame_q2

record = False


def recording_button(file_suffix):
    global record, \
        button_repeat_q2, button_repeat_image_q2, button_repeat_image_hover_q2,\
        button_record_q2, button_record_image_q2, button_record_image_hover_q2, \
        button_play_q2, button_play_image_q2, button_play_image_hover_q2, \
        button_next_q2, button_next_image_q2, button_next_image_hover_q2

    if record:
        stop()

        button_record_image_q2 = PhotoImage(file=relative_to_assets("button_record.png"))
        button_record_image_hover_q2 = PhotoImage(file=relative_to_assets("button_record_hover.png"))
        button_record_q2["image"] = button_record_image_hover_q2

        button_repeat_image_q2 = PhotoImage(file=relative_to_assets("button_repeat.png"))
        button_repeat_image_hover_q2 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
        button_repeat_q2["image"] = button_repeat_image_q2
        button_repeat_q2["state"] = "normal"

        button_play_image_q2 = PhotoImage(file=relative_to_assets("button_play.png"))
        button_play_image_hover_q2 = PhotoImage(file=relative_to_assets("button_play_hover.png"))
        button_play_q2["image"] = button_play_image_q2
        button_play_q2["state"] = "normal"

        button_next_image_q2 = PhotoImage(file=relative_to_assets("button_next.png"))
        button_next_image_hover_q2 = PhotoImage(file=relative_to_assets("button_next_hover.png"))
        button_next_q2["image"] = button_next_image_q2
        button_next_q2["state"] = "normal"

    else:
        start(file_suffix)

        button_record_image_q2 = PhotoImage(file=relative_to_assets("button_stop_recording.png"))
        button_record_image_hover_q2 = PhotoImage(file=relative_to_assets("button_stop_recording_hover.png"))
        button_record_q2["image"] = button_record_image_hover_q2

        button_repeat_image_q2 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_image_hover_q2 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_q2["image"] = button_repeat_image_q2
        button_repeat_q2["state"] = "disabled"

        button_play_image_q2 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_image_hover_q2 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_q2["image"] = button_play_image_q2
        button_play_q2["state"] = "disabled"

        button_next_image_q2 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_image_hover_q2 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_q2["image"] = button_next_image_q2
        button_next_q2["state"] = "disabled"

    record = not record


def first_ask_q2():
    global \
        button_repeat_q2, button_repeat_image_q2, button_repeat_image_hover_q2, \
        button_record_q2, button_record_image_q2, button_record_image_hover_q2

    play_audio('ask_questions_speech/ask_q2.wav')

    button_repeat_image_q2 = PhotoImage(file=relative_to_assets("button_repeat.png"))
    button_record_image_q2 = PhotoImage(file=relative_to_assets("button_record.png"))

    button_repeat_image_hover_q2 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
    button_record_image_hover_q2 = PhotoImage(file=relative_to_assets("button_record_hover.png"))

    button_repeat_q2["image"] = button_repeat_image_q2
    button_record_q2["image"] = button_record_image_q2

    button_repeat_q2["state"] = "normal"
    button_record_q2["state"] = "normal"
