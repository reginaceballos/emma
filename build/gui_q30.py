from utils_gui import *
import tkinter as tk
from tkinter import Canvas, PhotoImage, Label, Button, Frame
import threading
import cv2
from PIL import Image, ImageTk
from tkvideo.tkvideo import tkvideo


width = 479
height = 269


def button_repeat_hover_q30(e):
    button_repeat_q30.config(
        image=button_repeat_image_hover_q30
    )
def button_repeat_leave_q30(e):
    button_repeat_q30.config(
        image=button_repeat_image_q30
    )

def button_record_hover_q30(e):
    button_record_q30.config(
        image=button_record_image_hover_q30
    )
def button_record_leave_q30(e):
    button_record_q30.config(
        image=button_record_image_q30
    )

def button_play_hover_q30(e):
    button_play_q30.config(
        image=button_play_image_hover_q30
    )
def button_play_leave_q30(e):
    button_play_q30.config(
        image=button_play_image_q30
    )

def button_next_hover_q30(e):
    button_next_q30.config(
        image=button_next_image_hover_q30
    )
def button_next_leave_q30(e):
    button_next_q30.config(
        image=button_next_image_q30
    )

def create_gui_q30(window, next_frame, next_first_ask_function):
    global \
    image_background_q30, image_file_background_q30, \
    image_header_q30, image_file_header_q30, \
    image_question_q30, image_file_question_q30, \
    image_no_video_display_q30, image_file_no_video_display_q30, \
    button_repeat_q30, button_repeat_image_q30, button_repeat_image_hover_q30, \
    button_record_q30, button_record_image_q30, button_record_image_hover_q30, \
    button_play_q30, button_play_image_q30, button_play_image_hover_q30, \
    button_next_q30, button_next_image_q30, button_next_image_hover_q30, \
    cap_play_q30, image_video_q30

    frame_q30 = Frame(window,
                     height=800,
                     width=1400)
    frame_q30.place(x=0, y=0)


    canvas_q30 = Canvas(
        frame_q30,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_q30.place(x=0, y=0)

    image_file_background_q30 = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_q30 = Label(
        frame_q30,
        image=image_file_background_q30,
        bd=0
    )
    image_background_q30.place(
        x=0,
        y=0,
    )

    image_file_header_q30 = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_q30 = Label(
        frame_q30,
        image=image_file_header_q30,
        bd=0
    )
    image_header_q30.place(
        x=0,
        y=0,
    )


    image_file_question_q30 = PhotoImage(
        file=relative_to_assets("image_question_q30.png"))

    image_question_q30 = Label(
        frame_q30,
        image=image_file_question_q30,
        bd=0
    )

    image_question_q30.place(
        x=0,
        y=163,
    )

    button_repeat_image_q30 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))
    button_repeat_q30 = Button(
        frame_q30,
        image=button_repeat_image_q30,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: play_audio('ask_questions_speech/ask_q30.wav'),
        relief="flat",
        state="disabled"
    )
    button_repeat_q30.place(
        x=175.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_repeat_image_hover_q30 = PhotoImage(
        file=relative_to_assets("button_repeat_disabled.png"))

    button_repeat_q30.bind('<Enter>', button_repeat_hover_q30)
    button_repeat_q30.bind('<Leave>', button_repeat_leave_q30)

    button_record_image_q30 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))
    button_record_q30 = Button(
        frame_q30,
        image=button_record_image_q30,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: recording_button('q30'),
        relief="flat"
    )
    button_record_q30.place(
        x=486.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_record_image_hover_q30 = PhotoImage(
        file=relative_to_assets("button_record_disabled.png"))

    button_record_q30.bind('<Enter>', button_record_hover_q30)
    button_record_q30.bind('<Leave>', button_record_leave_q30)

    button_play_image_q30 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))
    button_play_q30 = Button(
        frame_q30,
        image=button_play_image_q30,
        highlightthickness=0,
        # command=lambda: play_audio("answer_questions_speech/recording_q30.wav"),
        command=replay_video_and_audio_q30,
        relief="flat",
        state="disabled"
    )
    button_play_q30.place(
        x=797.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_play_image_hover_q30 = PhotoImage(
        file=relative_to_assets("button_play_disabled.png"))

    button_play_q30.bind('<Enter>', button_play_hover_q30)
    button_play_q30.bind('<Leave>', button_play_leave_q30)

    button_next_image_q30 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))
    button_next_q30 = Button(
        frame_q30,
        image=button_next_image_q30,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: move_to_next_question_frame(next_frame, next_first_ask_function),
        relief="flat",
        state="disabled"
    )
    button_next_q30.place(
        x=1108.0,
        y=520.0,
        width=171.0,
        height=171.0
    )

    button_next_image_hover_q30 = PhotoImage(
        file=relative_to_assets("button_next_disabled.png"))

    button_next_q30.bind('<Enter>', button_next_hover_q30)
    button_next_q30.bind('<Leave>', button_next_leave_q30)

    
    image_file_no_video_display_q30 = PhotoImage(
        file=relative_to_assets("image_no_video_display.png"))

    image_no_video_display_q30 = Label(
        frame_q30,
        image=image_file_no_video_display_q30,
        bd=0
    )

    image_no_video_display_q30.place(
        x=829,
        y=182,
    )



    image_frame_q30 = tk.Frame(frame_q30, width=width, height=height)
    image_frame_q30.place(
        x=829,
        y=182,
    )

    image_video_q30 = Label(image_frame_q30, width=width, height=height,
        image=image_file_no_video_display_q30,
        bd=0)
    image_video_q30.place(
        x=0,
        y=0
    )

    
    video_path = '/Users/reginaceballos/Documents/MIT/2024-02 - Spring/6.8510 Intelligent Multimodal Interfaces/Final Project/emma/answer_questions_video/video_q30.mp4'

    cap_replay_q30 = cv2.VideoCapture(video_path)




    return frame_q30

recording = False


def recording_button(file_suffix):
    global recording, \
        button_repeat_q30, button_repeat_image_q30, button_repeat_image_hover_q30,\
        button_record_q30, button_record_image_q30, button_record_image_hover_q30, \
        button_play_q30, button_play_image_q30, button_play_image_hover_q30, \
        button_next_q30, button_next_image_q30, button_next_image_hover_q30, \
        cap_play_q30, thread

    if not recording:

        recording = not recording

        start_audio(file_suffix)

        cap_play_q30 = cv2.VideoCapture(0)
        
        show_frame_q30()

        thread = threading.Thread(target=lambda: start_recording_video('q30'))
        thread.start()
        

        button_record_image_q30 = PhotoImage(file=relative_to_assets("button_stop_recording.png"))
        button_record_image_hover_q30 = PhotoImage(file=relative_to_assets("button_stop_recording_hover.png"))
        button_record_q30["image"] = button_record_image_hover_q30

        button_repeat_image_q30 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_image_hover_q30 = PhotoImage(file=relative_to_assets("button_repeat_disabled.png"))
        button_repeat_q30["image"] = button_repeat_image_q30
        button_repeat_q30["state"] = "disabled"

        button_play_image_q30 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_image_hover_q30 = PhotoImage(file=relative_to_assets("button_play_disabled.png"))
        button_play_q30["image"] = button_play_image_q30
        button_play_q30["state"] = "disabled"

        button_next_image_q30 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_image_hover_q30 = PhotoImage(file=relative_to_assets("button_next_disabled.png"))
        button_next_q30["image"] = button_next_image_q30
        button_next_q30["state"] = "disabled"


    else:
        recording = not recording
        stop_audio()
        stop_frame_q30()

        button_record_image_q30 = PhotoImage(file=relative_to_assets("button_record.png"))
        button_record_image_hover_q30 = PhotoImage(file=relative_to_assets("button_record_hover.png"))
        button_record_q30["image"] = button_record_image_hover_q30

        button_repeat_image_q30 = PhotoImage(file=relative_to_assets("button_repeat.png"))
        button_repeat_image_hover_q30 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
        button_repeat_q30["image"] = button_repeat_image_q30
        button_repeat_q30["state"] = "normal"

        button_play_image_q30 = PhotoImage(file=relative_to_assets("button_play.png"))
        button_play_image_hover_q30 = PhotoImage(file=relative_to_assets("button_play_hover.png"))
        button_play_q30["image"] = button_play_image_q30
        button_play_q30["state"] = "normal"

        button_next_image_q30 = PhotoImage(file=relative_to_assets("button_next.png"))
        button_next_image_hover_q30 = PhotoImage(file=relative_to_assets("button_next_hover.png"))
        button_next_q30["image"] = button_next_image_q30
        button_next_q30["state"] = "normal"



def first_ask_q30():
    global \
        button_repeat_q30, button_repeat_image_q30, button_repeat_image_hover_q30, \
        button_record_q30, button_record_image_q30, button_record_image_hover_q30

    play_audio('ask_questions_speech/ask_q30.wav')

    button_repeat_image_q30 = PhotoImage(file=relative_to_assets("button_repeat.png"))
    button_record_image_q30 = PhotoImage(file=relative_to_assets("button_record.png"))

    button_repeat_image_hover_q30 = PhotoImage(file=relative_to_assets("button_repeat_hover.png"))
    button_record_image_hover_q30 = PhotoImage(file=relative_to_assets("button_record_hover.png"))

    button_repeat_q30["image"] = button_repeat_image_q30
    button_record_q30["image"] = button_record_image_q30

    button_repeat_q30["state"] = "normal"
    button_record_q30["state"] = "normal"


def show_frame_q30():
    ret_play, frame_play = cap_play_q30.read()

    if ret_play:
        frame_play = cv2.resize(frame_play, (width, height))
        frame_pic = cv2.flip(frame_play, 1)
        cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        image_video_q30.imgtk = imgtk
        image_video_q30.configure(image=imgtk)
        image_video_q30.after(10, show_frame_q30)


def stop_frame_q30():
    if cap_play_q30.isOpened():
        cap_play_q30.release()

    image_video_q30.configure(image=image_file_no_video_display_q30)


def start_recording_video(file_suffix):
    global recording, cap_replay_q30

    video_path = 'answer_questions_video/video_answer_' + file_suffix + '.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('answer_questions_video/video_answer_' + file_suffix + '.mp4',fourcc,  25.0, (1280, 720))
    out = cv2.VideoWriter(video_path,fourcc,  25.0, (1280, 720))

    cap_record = cv2.VideoCapture(0)

    if not cap_record.isOpened():
        print("Error: Could not open camera.")
        return


    while recording:
        ret_record, frame_record = cap_record.read()
        
        if ret_record:
            out.write(frame_record)
        else:
            break

    cap_record.release()
    out.release()

    cap_replay_q30 = cv2.VideoCapture(video_path)


def replay_video_and_audio_q30():
    thread = threading.Thread(target=lambda: play_audio("answer_questions_speech/recording_q30.wav"))
    thread.start()
    time.sleep(1)
    replay_video_q30()


def replay_video_q30():

    ret_replay, frame_replay = cap_replay_q30.read()

    if ret_replay:
        frame_replay = cv2.resize(frame_replay, (width, height))

        frame_pic = cv2.flip(frame_replay, 1)

        cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        

        image_video_q30.imgtk = imgtk
        image_video_q30.configure(image=imgtk)
        image_video_q30.after(2, replay_video_q30)
    else:
        image_video_q30.configure(image=image_file_no_video_display_q30)
        cap_replay_q30.set(cv2.CAP_PROP_POS_FRAMES, 0)

