# import tkinter
# import cv2
# import PIL.Image, PIL.ImageTk
# import time

# class App:
#     def __init__(self, window, window_title, video_source=0):
#         self.window = window
#         self.window.title(window_title)
#         self.video_source = video_source

#         # open video source (by default this will try to open the computer webcam)
#         self.vid = MyVideoCapture(self.video_source)

#         # Create a canvas that can fit the above video source size
#         self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
#         self.canvas.pack()

#         # Button that lets the user take a snapshot
#         self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
#         self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

#         # After it is called once, the update method will be automatically called every delay milliseconds
#         self.delay = 15
#         self.update()

#         self.window.mainloop()

#     def snapshot(self):
#         # Get a frame from the video source
#         ret, frame = self.vid.get_frame()

#         if ret:
#             cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     def update(self):
#         # Get a frame from the video source
#         ret, frame = self.vid.get_frame()

#         if ret:
#             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
#             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

#         self.window.after(self.delay, self.update)


# class MyVideoCapture:
#     def __init__(self, video_source=0):
#         # Open the video source
#         self.vid = cv2.VideoCapture(video_source)
#         if not self.vid.isOpened():
#             raise ValueError("Unable to open video source", video_source)

#         # Get video source width and height
#         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

#     def get_frame(self):
#         if self.vid.isOpened():
#             ret, frame = self.vid.read()
#             if ret:
#                 # Return a boolean success flag and the current frame converted to BGR
#                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             else:
#                 return (ret, None)
#         else:
#             return (ret, None)

#     # Release the video source when the object is destroyed
#     def __del__(self):
#         if self.vid.isOpened():
#             self.vid.release()

# # Create a window and pass it to the Application object
# App(tkinter.Tk(), "Tkinter and OpenCV")




import multiprocessing
import tkinter as tk
import cv2

e = multiprocessing.Event()
p = None

# -------begin capturing and saving video
def startrecording(e):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4',fourcc,  25.0, (1280, 720))

    while(cap.isOpened()):
        if e.is_set():
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            e.clear()
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
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

    root.quit()
    root.destroy()

if __name__ == "__main__":
    # -------configure window
    root = tk.Tk()
    root.geometry("%dx%d+0+0" % (100, 100))
    startbutton=tk.Button(root,width=10,height=1,text='START',command=start_recording_proc)
    stopbutton=tk.Button(root,width=10,height=1,text='STOP', command=stoprecording)
    startbutton.pack()
    stopbutton.pack()

    # -------begin
    root.mainloop()




# import numpy as np
# import os
# import cv2


# filename = 'video.mp4'
# frames_per_second = 24.0
# res = '720p'

# # Set resolution for the video capture
# # Function adapted from https://kirr.co/0l6qmh
# def change_res(cap, width, height):
#     cap.set(3, width)
#     cap.set(4, height)

# # Standard Video Dimensions Sizes
# STD_DIMENSIONS =  {
#     "480p": (640, 480),
#     "720p": (1280, 720),
#     "1080p": (1920, 1080),
#     "4k": (3840, 2160),
# }


# # grab resolution dimensions and set video capture to it.
# def get_dims(cap, res='1080p'):
#     width, height = STD_DIMENSIONS["480p"]
#     if res in STD_DIMENSIONS:
#         width,height = STD_DIMENSIONS[res]
#     ## change the current caputre device
#     ## to the resulting resolution
#     change_res(cap, width, height)
#     return width, height

# # # Video Encoding, might require additional installs
# # # Types of Codes: http://www.fourcc.org/codecs.php
# # VIDEO_TYPE = {
# #     'avi': cv2.VideoWriter_fourcc(*'XVID'),
# #     #'mp4': cv2.VideoWriter_fourcc(*'H264'),
# #     'mp4': cv2.VideoWriter_fourcc(*'XVID'),
# # }

# # def get_video_type(filename):
# #     filename, ext = os.path.splitext(filename)
# #     if ext in VIDEO_TYPE:
# #       return  VIDEO_TYPE[ext]
# #     return VIDEO_TYPE['avi']



# cap = cv2.VideoCapture(0)

# # out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))
# out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 25, get_dims(cap, res))

# while True:
#     ret, frame = cap.read()
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# out.release()
# cv2.destroyAllWindows()




# import numpy as np
# import os
# import cv2


# filename = 'video.mp4'
# frames_per_second = 24.0
# res = '720p'

# # Set resolution for the video capture
# # Function adapted from https://kirr.co/0l6qmh
# def change_res(cap, width, height):
#     cap.set(3, width)
#     cap.set(4, height)

# # Standard Video Dimensions Sizes
# STD_DIMENSIONS =  {
#     "480p": (640, 480),
#     "720p": (1280, 720),
#     "1080p": (1920, 1080),
#     "4k": (3840, 2160),
# }


# # grab resolution dimensions and set video capture to it.
# def get_dims(cap, res='1080p'):
#     width, height = STD_DIMENSIONS["480p"]
#     if res in STD_DIMENSIONS:
#         width,height = STD_DIMENSIONS[res]
#     ## change the current caputre device
#     ## to the resulting resolution
#     change_res(cap, width, height)
#     return width, height

# # Video Encoding, might require additional installs
# # Types of Codes: http://www.fourcc.org/codecs.php
# VIDEO_TYPE = {
#     'avi': cv2.VideoWriter_fourcc(*'XVID'),
#     #'mp4': cv2.VideoWriter_fourcc(*'H264'),
#     'mp4': cv2.VideoWriter_fourcc(*'XVID'),
# }

# def get_video_type(filename):
#     filename, ext = os.path.splitext(filename)
#     if ext in VIDEO_TYPE:
#       return  VIDEO_TYPE[ext]
#     return VIDEO_TYPE['avi']



# cap = cv2.VideoCapture(0)
# out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

# while True:
#     ret, frame = cap.read()
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# out.release()
# cv2.destroyAllWindows()



# # packages
# import cv2
# from cv2 import VideoWriter
# from cv2 import VideoWriter_fourcc

# # open the webcam video stream
# webcam = cv2.VideoCapture(0)

# # open output video file stream
# video = VideoWriter('webcam.mp4', VideoWriter_fourcc(*'XVID'), 25.0, (640, 480))

# # main loop
# while True:
#     # get the frame from the webcam
#     stream_ok, frame = webcam.read()
    
#     # if webcam stream is ok
#     if stream_ok:
#         # display current frame
#         cv2.imshow('Webcam', frame)
        
#         # write frame to the video file
#         video.write(frame)

#     # escape condition
#     if cv2.waitKey(1) & 0xFF == 27: break

# # clean ups
# cv2.destroyAllWindows()

# # release web camera stream
# webcam.release()

# # release video output file stream
# video.release()


# import multiprocessing
# import tkinter as tk
# import cv2

# e = multiprocessing.Event()
# p = None

# # -------begin capturing and saving video
# def startrecording(e):
#     cap = cv2.VideoCapture(0)
#     fourcc = cv2.VideoWriter.fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi',fourcc,  20.0, (640,480))

#     while(cap.isOpened()):
#         if e.is_set():
#             cap.release()
#             out.release()
#             cv2.destroyAllWindows()
#             e.clear()
#         ret, frame = cap.read()
#         if ret==True:
#             out.write(frame)
#         else:
#             break

# def start_recording_proc():
#     global p
#     p = multiprocessing.Process(target=startrecording, args=(e,))
#     p.start()

# # -------end video capture and stop tk
# def stoprecording():
#     e.set()
#     p.join()

#     root.quit()
#     root.destroy()

# if __name__ == "__main__":
#     # -------configure window
#     root = tk.Tk()
#     root.geometry("%dx%d+0+0" % (100, 100))
#     startbutton=tk.Button(root,width=10,height=1,text='START',command=start_recording_proc)
#     stopbutton=tk.Button(root,width=10,height=1,text='STOP', command=stoprecording)
#     startbutton.pack()
#     stopbutton.pack()

#     # -------begin
#     root.mainloop()



# import numpy as np
# import cv2
# import tkinter as tk
# from PIL import Image, ImageTk

# #Set up GUI
# window = tk.Tk()  #Makes main window
# window.wm_title("Digital Microscope")
# window.config(background="#FFFFFF")

# #Graphics window
# imageFrame = tk.Frame(window, width=600, height=500)
# imageFrame.grid(row=0, column=0, padx=10, pady=2)

# #Capture video frames
# lmain = tk.Label(imageFrame)
# lmain.grid(row=0, column=0)
# cap = cv2.VideoCapture(0)
# def show_frame():
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(10, show_frame) 



# #Slider window (slider controls stage position)
# sliderFrame = tk.Frame(window, width=600, height=100)
# sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


# show_frame()  #Display 2
# window.mainloop()  #Starts GUI