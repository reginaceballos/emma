from utils_gui import *
from tkinter import Canvas, PhotoImage, Label, Button, Frame, Entry, font, StringVar


def button_login_hover(e):
    button_login.config(
        image=button_login_image_hover
    )
def button_login_leave(e):
    button_login.config(
        image=button_login_image
    )

def create_gui_healthworker(window, next_frame):
    global \
    image_background_healthworker, image_file_background_healthworker, \
    image_header_healthworker, image_file_header_healthworker, \
    image_question_healthworker, image_file_question_healthworker, \
    image_welcome_healthworker, image_file_welcome_healthworker, \
    image_wrong_password_healthworker, image_file_wrong_password_healthworker, \
    button_login, button_login_image, button_login_image_hover, \
    username, username_sv, password, password_sv

    frame_healthworker = Frame(window,
                     height=800,
                     width=1400)
    frame_healthworker.place(x=0, y=0)


    canvas_healthworker = Canvas(
        frame_healthworker,
        bg="#FFFFFF",
        height=800,
        width=1400,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas_healthworker.place(x=0, y=0)

    image_file_background_healthworker = PhotoImage(
        file=relative_to_assets("image_background.png"))

    image_background_healthworker = Label(
        frame_healthworker,
        image=image_file_background_healthworker,
        bd=0
    )
    image_background_healthworker.place(
        x=0,
        y=0,
    )

    image_file_header_healthworker = PhotoImage(
        file=relative_to_assets("image_header.png"))

    image_header_healthworker = Label(
        frame_healthworker,
        image=image_file_header_healthworker,
        bd=0
    )
    image_header_healthworker.place(
        x=0,
        y=0,
    )


    image_file_question_healthworker = PhotoImage(
        file=relative_to_assets("image_question_healthworker.png"))

    image_question_healthworker = Label(
        frame_healthworker,
        image=image_file_question_healthworker,
        bd=0
    )

    image_question_healthworker.place(
        x=0,
        y=228,
    )

    image_file_welcome_healthworker = PhotoImage(
        file=relative_to_assets("image_welcome_healthworker.png"))

    image_welcome_healthworker = Label(
        frame_healthworker,
        image=image_file_welcome_healthworker,
        bd=0
    )

    image_welcome_healthworker.place(
        x=915,
        y=163,
    )


    username_sv = StringVar()
    password_sv = StringVar()

    username_sv.trace_add("write", activate_login)
    password_sv.trace_add("write", activate_login)


    entry_username_healthworker = Entry(
        frame_healthworker,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        selectforeground="#2F1893",
        font =('DM Sans',24),
        textvariable=username_sv
    )
    entry_username_healthworker.place(
        x=955.0,
        y=286.0,
        width=291.0,
        height=65.0
    )


    entry_password_healthworker = Entry(
        frame_healthworker,
        bd=0,
        bg="#7999D4",
        fg="#FFFFFF",
        highlightthickness=0,
        selectforeground="#2F1893",
        show="●",
        font =('DM Sans',24),
        textvariable=password_sv
    )
    entry_password_healthworker.place(
        x=955.0,
        y=420.0,
        width=291.0,
        height=66.0
    )



    button_login_image = PhotoImage(
        file=relative_to_assets("button_login_disabled.png"))
    button_login = Button(
        frame_healthworker,
        image=button_login_image,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: login_to_next_page(next_frame),
        relief="flat",
        state="disabled"
    )

    button_login.place(
        x=921.0,
        y=569.0,
        width=359.0,
        height=68.0
    )

    button_login_image_hover = PhotoImage(
        file=relative_to_assets("button_login_disabled.png"))

    button_login.bind('<Enter>', button_login_hover)
    button_login.bind('<Leave>', button_login_leave)


    image_file_wrong_password_healthworker = PhotoImage(
        file=relative_to_assets("image_wrong_password_disabled.png"))

    image_wrong_password_healthworker = Label(
        frame_healthworker,
        image=image_file_wrong_password_healthworker,
        bd=0
    )
    image_wrong_password_healthworker.place(
        x=915,
        y=532,
    )
    




    return frame_healthworker




def activate_login(*args):
    global \
        button_login, button_login_image, button_login_image_hover, \
        image_wrong_password_healthworker, image_file_wrong_password_healthworker


    image_file_wrong_password_healthworker = PhotoImage(file=relative_to_assets("image_wrong_password_disabled.png"))
    
    image_wrong_password_healthworker["image"] = image_file_wrong_password_healthworker



    if username_sv.get() != "" and password_sv.get() != "":
        button_login_image = PhotoImage(file=relative_to_assets("button_login.png"))

        button_login_image_hover = PhotoImage(file=relative_to_assets("button_login_hover.png"))

        button_login["image"] = button_login_image

        button_login["state"] = "normal"
    else:
        button_login_image = PhotoImage(file=relative_to_assets("button_login_disabled.png"))

        button_login_image_hover = PhotoImage(file=relative_to_assets("button_login_disabled.png"))

        button_login["image"] = button_login_image

        button_login["state"] = "disabled"


def login_to_next_page(next_frame, *args):
    global \
        image_wrong_password_healthworker, image_file_wrong_password_healthworker
    
    if username_sv.get() == 'emma' and password_sv.get() == '1234':
        image_file_wrong_password_healthworker = PhotoImage(file=relative_to_assets("image_wrong_password_disabled.png"))
    
        image_wrong_password_healthworker["image"] = image_file_wrong_password_healthworker

        move_to_next_question_frame(next_frame)


    else:
        image_file_wrong_password_healthworker = PhotoImage(file=relative_to_assets("image_wrong_password.png"))
    
        image_wrong_password_healthworker["image"] = image_file_wrong_password_healthworker
        