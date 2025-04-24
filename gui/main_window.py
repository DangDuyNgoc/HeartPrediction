from tkinter import PhotoImage, Label, Button
from gui.info_window import Info
from gui.detail_entry import create_detail_entry
from gui.graph_handler import handleAnalysis
from logic.model_handler import logout, Save

def create_main_window(root):
    root.title("Heart Attack Prediction")
    root.geometry("1450x800+40+20")
    root.resizable(False, False)
    root.config(bg="#f0ddd5")

    # Icon and Header
    image_icon = PhotoImage(file='images/icon.png')
    root.iconphoto(False, image_icon)

    logo = PhotoImage(file='images/header.png')
    header = Label(image=logo, bg="#f0ddd5")
    header.place(x=0, y=0)

    # Buttons
    info_button = PhotoImage(file='images/info.png')
    Button(root, image=info_button, cursor="hand2", bd=0, background="#f0ddd5", command=Info).place(x=10, y=240)

    analysis_button = PhotoImage(file='images/Analysis.png').subsample(2)
    Button(root, image=analysis_button, bd=0, bg="#f0ddd5", cursor='hand2', command=handleAnalysis).place(x=1200, y=260)

    save_button = PhotoImage(file='images/save.png')
    Button(root, image=save_button, cursor="hand2", bd=0, background="#f0ddd5", command=Save).place(x=1320, y=260)

    logout_button = PhotoImage(file='images/logout_icon.png').subsample(2)
    Button(root, image=logout_button, bg="#f0ddd5", bd=0, cursor="hand2", command=logout).place(x=1370, y=220)

    # Detail Entry Section
    create_detail_entry(root)