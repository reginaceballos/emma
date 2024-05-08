from tkinter import Tk
from build.gui_healthworker import *
from build.gui_results import *
from utils_gui import *


def start_screen():
    window = Tk()

    window.geometry("1400x800")
    window.title("EMMA")
    window.configure(bg="#FFFFFF")

    frame_results = create_gui_results(window)
    frame_results.grid(row=0, column=0)

    frame_healthworker = create_gui_healthworker(window, frame_results)
    frame_healthworker.grid(row=0, column=0)

    window.mainloop()


start_screen()
