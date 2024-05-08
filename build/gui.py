
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, font


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/reginaceballos/Documents/MIT/2024-02 - Spring/6.8510 Intelligent Multimodal Interfaces/Final Project/emma/build/assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1400x800")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 800,
    width = 1400,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    700.0,
    400.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    700.0,
    470.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    700.0,
    36.0,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    339.0,
    340.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    1052.0,
    479.0,
    image=image_image_5
)

canvas.create_text(
    965.0,
    409.0,
    anchor="nw",
    text="Embarrassment",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    965.0,
    633.0,
    anchor="nw",
    text="Joy",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    965.0,
    313.0,
    anchor="nw",
    text="Disappointment",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    965.0,
    537.0,
    anchor="nw",
    text="Sadness",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    965.0,
    361.0,
    anchor="nw",
    text="Empathic Pain",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    965.0,
    585.0,
    anchor="nw",
    text="Disgust",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    107.0,
    389.0,
    anchor="nw",
    text="Patient is not at risk of depression",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    324.0,
    462.0,
    anchor="nw",
    text="80%",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    1230.0,
    361.0,
    anchor="nw",
    text="80%",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    1230.0,
    585.0,
    anchor="nw",
    text="80%",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    1230.0,
    409.0,
    anchor="nw",
    text="80%",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    1230.0,
    633.0,
    anchor="nw",
    text="80%",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    1230.0,
    313.0,
    anchor="nw",
    text="80%",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

canvas.create_text(
    1230.0,
    537.0,
    anchor="nw",
    text="80%",
    fill="#FFFFFF",
    font=("DMSans Bold", 32 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=176.0,
    y=635.0,
    width=359.0,
    height=68.0
)

button_image_hover_1 = PhotoImage(
    file=relative_to_assets("button_hover_1.png"))

def button_1_hover(e):
    button_1.config(
        image=button_image_hover_1
    )
def button_1_leave(e):
    button_1.config(
        image=button_image_1
    )

button_1.bind('<Enter>', button_1_hover)
button_1.bind('<Leave>', button_1_leave)

#window.resizable(False, False)
window.mainloop()
