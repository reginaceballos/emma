import multiprocessing
import tkinter as tk
from tkinter import PhotoImage, Label
import cv2
from PIL import Image, ImageTk

e = multiprocessing.Event()
p = None

# -------begin capturing and saving video
def startrecording(e):
    cap_record = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4',fourcc,  25.0, (1280, 720))

    while(cap_record.isOpened()):
        if e.is_set():
            cap_record.release()
            out.release()
            cv2.destroyAllWindows()
            e.clear()
        ret_record, frame_record = cap_record.read()
        
        if ret_record:
            out.write(frame_record)
        else:
            break

def start_recording_proc():
    global p
    p = multiprocessing.Process(target=startrecording, args=(e,))
    p.start()

# -------end video capture and stop tk
def stoprecording():
    e.set()
    p.join()

    # window.quit()
    # window.destroy()


def show_frame():
    ret_play, frame_play = cap_play.read()

    if ret_play:
        frame_pic = cv2.flip(frame_play, 1)
        cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        image_video.imgtk = imgtk
        image_video.configure(image=imgtk)
        image_video.after(10, show_frame)


global window

if __name__ == "__main__":
    # -------configure window
    window = tk.Tk()
    # window.geometry("%dx%d+0+0" % (100, 100))

    window.title('Record video')


    width = cv2.VideoCapture(0).get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cv2.VideoCapture(0).get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # canvas = tk.Canvas(window, width = width, height = height)
    # canvas.pack()

    imageFrame = tk.Frame(window, width=600, height=500)
    # imageFrame.grid(row=0, column=0, padx=10, pady=2)
    imageFrame.pack()

    # image_file_video = PhotoImage(
    #     file="build/assets/frames/image_welcome_healthworker.png")

    image_video = Label(
        # window,
        # image=image_file_video,
        # bd=0,
        imageFrame,
    )
    image_video.pack()
    # image_video.grid(row=0, column=0)
    cap_play = cv2.VideoCapture(0)

    show_frame()


    startbutton=tk.Button(window,width=10,height=1,text='START',command=start_recording_proc)
    stopbutton=tk.Button(window,width=10,height=1,text='STOP', command=stoprecording)
    startbutton.pack()
    stopbutton.pack()

    # -------begin
    window.mainloop()





    

