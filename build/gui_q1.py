from utils_gui import *
import tkinter as tk
from tkinter import Canvas, PhotoImage, Label, Button, Frame
import threading
#from video_recorder import *
import cv2
from PIL import Image, ImageTk


width = 479
height = 269


def button_repeat_hover_q1(e):
    button_repeat_q1.config(
        image=button_repeat_image_hover_q1
    )
def button_repeat_leave_q1(e):
    button_repeat_q1.config(
        image=button_repeat_image_q1
    )

def button_record_hover_q1(e):
    button_record_q1.config(
        image=button_record_image_hover_q1
    )
def button_record_leave_q1(e):
    button_record_q1.config(
        image=button_record_image_q1
    )

def button_play_hover_q1(e):
    button_play_q1.config(
        image=button_play_image_hover_q1
    )
def button_play_leave_q1(e):
    button_play_q1.config(
        image=button_play_image_q1
    )

def button_next_hover_q1(e):
    button_next_q1.config(
        image=button_next_image_hover_q1
    )
def button_next_leave_q1(e):
    button_next_q1.config(
        image=button_next_image_q1
    )

def create_gui_q1(window, next_frame, next_first_ask_function):
    global \
    image_background_q1, image_file_background_q1, \
    image_header_q1, image_file_header_q1, \
    image_question_q1, image_file_question_q1, \
    image_no_video_display_q1, image_file_no_video_display_q1, \
    button_repeat_q1, button_repeat_image_q1, button_repeat_image_hover_q1, \
    button_record_q1, button_record_image_q1, button_record_image_hover_q1, \
    button_play_q1, button_play_image_q1, button_play_image_hover_q1, \
    button_next_q1, button_next_image_q1, button_next_image_hover_q1, \
    cap_play_q1, image_video_q1

    frame_q1 = Frame(window,
                     height=800,
                     width=1400)
    frame_q1.place(x=0, y=0)


    canvas_q1 = Canvas(
        frame_q1,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_q1.place(x=0, y=0)

    image_file_background_q1 = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_q1 = Label(
        frame_q1,
        image=image_file_background_q1,
        bd=0
    )
    image_background_q1.place(
        x=0,
        y=0,
    )

    image_file_header_q1 = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_q1 = Label(
        frame_q1,
        image=image_file_header_q1,
        bd=0
    )
    image_header_q1.place(
        x=0,
        y=0,
    )


    image_file_question_q1 = PhotoImage(
        file=relative_to_assets("image_question_q1.png"))

    image_question_q1 = Label(
        frame_q1,
        image=image_file_question_q1,
        bd=0
    )

    image_question_q1.place(
        x=0,
        y=210,
    )

    button_repeat_image_q1 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))
    button_repeat_q1 = Button(
        frame_q1,
        image=button_repeat_image_q1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: play_audio('ask_questions_speech/ask_q1.wav'),
        relief="flat",
        state="disabled"
    )
    button_repeat_q1.place(
        x=175.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_repeat_image_hover_q1 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))

    button_repeat_q1.bind('<Enter>', button_repeat_hover_q1)
    button_repeat_q1.bind('<Leave>', button_repeat_leave_q1)

    button_record_image_q1 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))
    button_record_q1 = Button(
        frame_q1,
        image=button_record_image_q1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: recording_button('q1'),
        relief="flat"
    )
    button_record_q1.place(
        x=486.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_record_image_hover_q1 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))

    button_record_q1.bind('<Enter>', button_record_hover_q1)
    button_record_q1.bind('<Leave>', button_record_leave_q1)

    button_play_image_q1 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))
    button_play_q1 = Button(
        frame_q1,
        image=button_play_image_q1,
        highlightthickness=0,
        command=lambda: play_audio("answer_questions_speech/recording_q1.wav"),
        relief="flat",
        state="disabled"
    )
    button_play_q1.place(
        x=797.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_play_image_hover_q1 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))

    button_play_q1.bind('<Enter>', button_play_hover_q1)
    button_play_q1.bind('<Leave>', button_play_leave_q1)

    button_next_image_q1 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))
    button_next_q1 = Button(
        frame_q1,
        image=button_next_image_q1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: move_to_next_question_frame(next_frame, next_first_ask_function),
        relief="flat",
        state="disabled"
    )
    button_next_q1.place(
        x=1108.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_next_image_hover_q1 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))

    button_next_q1.bind('<Enter>', button_next_hover_q1)
    button_next_q1.bind('<Leave>', button_next_leave_q1)

    
    image_file_no_video_display_q1 = PhotoImage(
        file=relative_to_assets("image_no_video_display.png"))

    image_no_video_display_q1 = Label(
        frame_q1,
        image=image_file_no_video_display_q1,
        bd=0
    )

    image_no_video_display_q1.place(
        x=829,
        y=182,
    )



    image_frame_q1 = tk.Frame(frame_q1, width=width, height=height)
    image_frame_q1.place(
        x=829,
        y=182,
        bd=0
    )

    image_video_q1 = Label(image_frame_q1, width=width, height=height)
    image_video_q1.place(
        x=0,
        y=0,
        bd=0
    )


    

    return frame_q1

record = False


def recording_button(file_suffix):
    global record, \
        button_repeat_q1, button_repeat_image_q1, button_repeat_image_hover_q1,\
        button_record_q1, button_record_image_q1, button_record_image_hover_q1, \
        button_play_q1, button_play_image_q1, button_play_image_hover_q1, \
        button_next_q1, button_next_image_q1, button_next_image_hover_q1, \
        cap_play_q1

    if record:
        # thread_1 = threading.Thread(target=stop)
        # thread_2 = threading.Thread(target=start_recording_proc)
        # thread_1.start()
        # thread_2.start()
        stop()
        stop_frame()

        button_record_image_q1 = PhotoImage(file=relative_to_assets("button_record.png"))
        button_record_image_hover_q1 = PhotoImage(file=relative_to_assets("button_record_hover.png"))
        button_record_q1["image"] = button_record_image_hover_q1

        button_repeat_image_q1 = PhotoImage(file=relative_to_assets("button_repeat.png"))
        button_repeat_image_hover_q1 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
        button_repeat_q1["image"] = button_repeat_image_q1
        button_repeat_q1["state"] = "normal"

        button_play_image_q1 = PhotoImage(file=relative_to_assets("button_play.png"))
        button_play_image_hover_q1 = PhotoImage(file=relative_to_assets("button_play_hover.png"))
        button_play_q1["image"] = button_play_image_q1
        button_play_q1["state"] = "normal"

        button_next_image_q1 = PhotoImage(file=relative_to_assets("button_next.png"))
        button_next_image_hover_q1 = PhotoImage(file=relative_to_assets("button_next_hover.png"))
        button_next_q1["image"] = button_next_image_q1
        button_next_q1["state"] = "normal"

    else:
        start(file_suffix)

        cap_play_q1 = cv2.VideoCapture(0)
        
        show_frame()

        button_record_image_q1 = PhotoImage(file=relative_to_assets("button_stop_recording.png"))
        button_record_image_hover_q1 = PhotoImage(file=relative_to_assets("button_stop_recording_hover.png"))
        button_record_q1["image"] = button_record_image_hover_q1

        button_repeat_image_q1 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_image_hover_q1 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_q1["image"] = button_repeat_image_q1
        button_repeat_q1["state"] = "disabled"

        button_play_image_q1 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_image_hover_q1 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_q1["image"] = button_play_image_q1
        button_play_q1["state"] = "disabled"

        button_next_image_q1 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_image_hover_q1 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_q1["image"] = button_next_image_q1
        button_next_q1["state"] = "disabled"

    record = not record



def first_ask_q1():
    global \
        button_repeat_q1, button_repeat_image_q1, button_repeat_image_hover_q1, \
        button_record_q1, button_record_image_q1, button_record_image_hover_q1

    play_audio('ask_questions_speech/ask_q1.wav')

    button_repeat_image_q1 = PhotoImage(file=relative_to_assets("button_repeat.png"))
    button_record_image_q1 = PhotoImage(file=relative_to_assets("button_record.png"))

    button_repeat_image_hover_q1 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
    button_record_image_hover_q1 = PhotoImage(file=relative_to_assets("button_record_hover.png"))

    button_repeat_q1["image"] = button_repeat_image_q1
    button_record_q1["image"] = button_record_image_q1

    button_repeat_q1["state"] = "normal"
    button_record_q1["state"] = "normal"


def show_frame():
    ret_play, frame_play = cap_play_q1.read()

    if ret_play:
        frame_play = cv2.resize(frame_play, (width, height))
        frame_pic = cv2.flip(frame_play, 1)
        cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        image_video_q1.imgtk = imgtk
        image_video_q1.configure(image=imgtk)
        image_video_q1.after(10, show_frame)


def stop_frame():
    if cap_play_q1.isOpened():
        cap_play_q1.release()

    image_video_q1.configure(image=image_file_no_video_display_q1)
