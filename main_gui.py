from tkinter import Tk
from build.gui_landing import *
from build.gui_q1 import *
from build.gui_q4 import *
from build.gui_q5 import *
from build.gui_thanks import *
from utils_gui import *

width = 479
height = 269

def show_frame():
    ret_play, frame_play = cap_play.read()

    if ret_play:
        frame_play = cv2.resize(frame_play, (width, height))
        frame_pic = cv2.flip(frame_play, 1)
        cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        image_video.imgtk = imgtk
        image_video.configure(image=imgtk)
        image_video.after(10, show_frame)


def start_screen():
    global frame_q2, cap_play, image_video
    window = Tk()

    window.geometry("1400x800")
    window.title("EMMA")
    window.configure(bg="#FFFFFF")

    frame_thanks = create_gui_thanks(window)
    frame_thanks.grid(row=0, column=0)

    frame_q5 = create_gui_q5(window, frame_thanks, first_ask_thanks)
    frame_q5.grid(row=0, column=0)

    frame_q4 = create_gui_q4(window, frame_q5, first_ask_q5)
    frame_q4.grid(row=0, column=0)

    frame_q1 = create_gui_q1(window, frame_q4, first_ask_q4)
    frame_q1.grid(row=0, column=0)

    frame_landing = create_gui_landing(window, frame_q1, first_ask_q1)
    frame_landing.grid(row=0, column=0)

    

    # imageFrame = tk.Frame(frame_q1, width=width, height=height)
    # imageFrame.place(
    #     x=829,
    #     y=182,
    # )

    # image_video = Label(imageFrame, width=width, height=height)
    # image_video.pack()
    
    # cap_play = cv2.VideoCapture(0)

    # show_frame()


    thread_speech(first_ask_landing)

    window.mainloop()


start_screen()