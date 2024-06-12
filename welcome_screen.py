from tkinter import *

def create_welcome_screen(root, background, framefg, start_callback):
    welcome_frame = Frame(root, bg=background)
    welcome_frame.place(x=0, y=0, width=1450, height=800)
    
    welcome_label = Label(welcome_frame, text="Welcome to Prediction Heart Disease", font=("Helvetica", 24, "bold"), bg=background, fg=framefg)
    welcome_label.pack(pady=200)
    
    start_button = Button(welcome_frame, text="Start", font=("Helvetica", 18), bg="#62a7ff", fg=framefg, command=start_callback)
    start_button.pack(pady=20)
    
    return welcome_frame
