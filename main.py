from tkinter import *
from datetime import date
from tkinter.ttk import Combobox
import datetime
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os

import matplotlib
matplotlib.use("TKAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from backend import *

background = "#f0ddd5"
framebg = "#ccc"
framefg= "#fefbfb"

root = Tk()
root.title("Heart Attack Prediction")
root.geometry("1450x800+40+20") #chiều rộng là 1450 pixel và chiều cao là 730 pixel 
 # 40 vị trí từ phía bên trái của màn hình và 20 chỉ vị trí từ phía trên .
root.resizable(False, False) #không thể kéo căn chỉnh kích thước của cửa sổ bằng chuột.
root.config(bg=background)

def Info():
    info_window = Toplevel(root)
    info_window.title("info")
    info_window.geometry("700x600+100+100")

    Label(info_window, text="Information related to dataset", font="robot 15 bold").pack(padx=20, pady=20)

    Label(info_window, text="age - age in years", font="arial 12").place(x=20, y=100)
    Label(info_window, text="sex - sex (1 = male; 0 = female)", font="arial 12").place(x=20, y=130)
    Label(info_window, text="cp - chest pain type(0 = typical angina; 1 = atypical; 2 = non-anginal pain; 3 = asymptomatic)", 
        font="arial 12").place(x=20, y=160)
    Label(info_window, text="trestbps - resting blood pressure(in mm Hg on to the hospital)", font="arial 12").place(x=20, y=190)
    Label(info_window, text="chol - serum cholesterol in mg/dl", font="arial 12").place(x=20, y=220)
    Label(info_window, text="fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", font="arial 12").place(x=20, y=250)
    Label(info_window, text="restecg - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)", font="arial 12").place(x=20, y=280)
    Label(info_window, text="thalach - maximum heart rate achieved", font="arial 12").place(x=20, y=310)
    Label(info_window, text="exang - exercise induced angina (1 = yes; 0 = no)", font="arial 12").place(x=20, y=340)
    Label(info_window, text="oldpeak - ST depression induced by exercise relative to rest", font="arial 12").place(x=20, y=370)
    Label(info_window, text="slope - the slope of the peak exercise ST segment(0 = upsloping; 1 = flat; 2 = downsloping )", font="arial 12").place(x=20, y=400)
    Label(info_window, text="ca - number of major vessels", font="arial 12").place(x=20, y=430)
    Label(info_window, text="thal - Thalassemia (0 = normal; 1 = fixed defect; 2 reversible defect)", font="arial 12").place(x=20, y=460)


    info_window.mainloop()

def logout():
    root.destroy()

def Clear():
    Name.get("")
    DayOfYear.get("")
    trestbps.get("")
    chol.get("")
    thalach.set("")
    oldpeak.set("")
    
Heading_entry = Frame(root, width=800, height=190, bg = "#df2d4b")
Heading_entry.config(highlightbackground="blue", highlightcolor="blue", highlightthickness=3)
Heading_entry.place(x=600, y = 20)

Label(Heading_entry, text="Registration No.", font="arial 15", bg= "#df2d4b", fg= framefg).place(x= 30, y =0)
Label(Heading_entry, text="Date", font="arial 15", bg= "#df2d4b", fg= framefg).place(x= 430, y =0)
Label(Heading_entry, text="Patient Name", font="arial 15", bg= "#df2d4b", fg= framefg).place(x= 30, y =90)
Label(Heading_entry, text="Birth Year", font="arial 15", bg= "#df2d4b", fg= framefg).place(x= 430, y = 90)

Registration = IntVar()
reg_entry = Entry(Heading_entry, textvariable=Registration, width=30, font= "arial 15", bg= "#ededed", fg= "#222222", bd= 0)
reg_entry.place(x=30, y =45)

Date = StringVar()
today = date.today()
d1 = today.strftime("%d/%m/%Y")
date_entry = Entry(Heading_entry, textvariable=Date, width=15, font= "arial 15", bg= "#ededed", fg= "#222222", bd= 0)
date_entry.place(x = 430, y = 45)
Date.set(d1)

Name = StringVar()
name_entry = Entry(Heading_entry, textvariable=Name, width=30, font= "arial 15", bg= "#ededed", fg= "#222222", bd= 0)
name_entry.place(x = 30, y = 130)

DayOfYear = IntVar()
year_entry = Entry(Heading_entry, textvariable=DayOfYear, width=30, font= "arial 15", bg= "#ededed", fg= "#222222", bd= 0)
year_entry.place(x = 430, y = 130)

Detail_entry = Frame(root, width = 490, height= 260, bg = "#dbe0e3")
Detail_entry.place(x = 30, y = 450)

Label(Detail_entry, text = "sex:", font = "arial 15", bg = framebg, fg = framefg).place(x = 10, y =10)
Label(Detail_entry, text = "fbs:", font = "arial 15", bg = framebg, fg = framefg).place(x = 180, y =10)
Label(Detail_entry, text = "exang:", font = "arial 15", bg = framebg, fg = framefg).place(x = 335, y =10)

def genSelection(): 
    if gen.get() == 1:
        Gender = 1
        return(Gender)
    elif gen.get() == 2:
        Gender = 0
        return(Gender)
    print(Gender)

def fbsSelection(): 
    if fbs.get() == 1:
        FBS = 1
        return(FBS)
    elif fbs.get() == 2:
        FBS = 0
        return(FBS)
    print(FBS)

def exangSelection(): 
    if exang.get() == 1:
        EXANG = 1
        return(EXANG)
    elif exang.get() == 2:
        EXANG = 0
        return(EXANG)
    print(EXANG)

gen = IntVar()
RMale = Radiobutton(Detail_entry, text= "Male", variable=gen, value= 1, command=genSelection)
RFe = Radiobutton(Detail_entry, text= "Female", variable=gen, value= 2, command=genSelection)
RMale.place(x = 43, y =10)
RFe.place(x = 93, y = 10)

fbs = IntVar() # đường huyết sau một thời gian nhịn ăn
RTrue = Radiobutton(Detail_entry, text= "True", variable=fbs, value= 1, command=fbsSelection)
RFalse = Radiobutton(Detail_entry, text= "False", variable=fbs, value= 2, command=fbsSelection)
RTrue.place(x = 213, y =10)
RFalse.place(x = 263, y = 10)

exang = IntVar() # tập thể dục gây đau thắt ngực Exercise induced angina
R1 = Radiobutton(Detail_entry, text= "Yes", variable=exang, value= 1, command=exangSelection)
R2 = Radiobutton(Detail_entry, text= "No", variable=exang, value= 2, command=exangSelection)
R1.place(x = 387, y =10)
R2.place(x = 430, y = 10)

Label(Detail_entry, text = "cp", font = "arial 15", bg = framebg, fg = framefg).place(x=10, y =50)
Label(Detail_entry, text = "restecg", font = "arial 15", bg = framebg, fg = framefg).place(x=10, y =90)
Label(Detail_entry, text = "slope", font = "arial 15", bg = framebg, fg = framefg).place(x=10, y =130)
Label(Detail_entry, text = "ca", font = "arial 15", bg = framebg, fg = framefg).place(x=10, y =170)
Label(Detail_entry, text = "thal", font = "arial 15", bg = framebg, fg = framefg).place(x=10, y =210)

def cpSelection():
    input = cp_combobox.get()
    if input == "0 = typical":
        return(0)
    elif input == "1 = atypical angina":
        return(1)
    elif input == "2 = non-anginal pain":
        return(2)
    elif input == "3 = asymptomatic":
        return(3)
    else:
        print()

def slopeSelection():
    input = slope_combobox.get()
    if input == "0 = upsloping":
        return(0)
    elif input == "1 = flat":
        return(1)
    elif input == "2 = downsloping":
        return(2)
    else:
        print()

cp_combobox = Combobox(Detail_entry, values= [
    '0 = typical', 
    '1 = atypical angina', 
    '2 = non-anginal pain',
    '3 = asymptomatic'
], font="arial 12", state="r", width=14)
restecg_combobox = Combobox(Detail_entry, values= ['0', '1', '2'], font="arial 12", state="r", width=11)
slope_combobox = Combobox(Detail_entry, values= [
    '0 = upsloping', 
    '1 = flat', 
    '2 = downsloping'
], font="arial 12", state="r", width=12)
ca_combobox = Combobox(Detail_entry, values= ['0', '1', '2', '3', '4'], font="arial 12", state="r", width=14)
thal_combobox = Combobox(Detail_entry, values= ['0', '1', '2', '3'], font="arial 12", state="r", width=14)

cp_combobox.place(x=50, y=50)
restecg_combobox.place(x=80, y=90)
slope_combobox.place(x=70, y=130)
ca_combobox.place(x=50, y=170)
thal_combobox.place(x=50, y=210)

Label(Detail_entry, text="Smoking", font="arial 13", width=7, bg=framebg, fg="black").place(x=240, y = 50)
Label(Detail_entry, text="trestbps", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=240, y = 90)
Label(Detail_entry, text="chol", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=240, y = 130)
Label(Detail_entry, text="thalach", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=240, y = 170)
Label(Detail_entry, text="odlpeak", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=240, y = 210)

trestbps = StringVar()
chol = StringVar()
thalach = StringVar()
oldpeak = StringVar()

trestbps_entry = Entry(Detail_entry, textvariable=trestbps, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)
chol_entry = Entry(Detail_entry, textvariable=chol, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)
thalach_entry = Entry(Detail_entry, textvariable=thalach, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)
oldpeak_entry = Entry(Detail_entry, textvariable=oldpeak, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)

trestbps_entry.place(x=320, y=90)
chol_entry.place(x=320, y=130)
thalach_entry.place(x=320, y=170)
oldpeak_entry.place(x=320, y=210)

report = Label(root, text="Hello world", font="arial 20 bold", bg="white", fg="#8dc63f")
report.place(x=1170, y=550)

report1 = Label(root, text="Hello world", font="arial 10 bold", bg="white")
report1.place(x=1130, y=610)

def handleAnalysis():
    name = Name.get()
    D1 = Date.get()
    today = datetime.date.today()
    A = today.year - DayOfYear.get()

    try:
        genChoice = genSelection()
    except:
        messagebox.showerror("Error", "Please select your gender!")
        return
    try:
        fbsChoice = fbsSelection()
    except:
        messagebox.showerror("Error", "Please select fbs!")
        return
    try:
        exangChoice = exangSelection()
    except:
        messagebox.showerror("Error", "Please select exang!")
        return
    try:
        cpChoice = int(cpSelection())
    except:
        messagebox.showerror("Error", "Please select cp!")
        return
    try:
        resChoice = int(restecg_combobox.get())
    except:
        messagebox.showerror("Error", "Please select restcg!!")
        return
    try:
        slopeChoice = int(slopeSelection())
    except:
        messagebox.showerror("Error", "Please select slope!")
        return
    try:
        caChoice = int(ca_combobox.get())
    except:
        messagebox.showerror("Error", "Please select ca!")
        return
    try:
        thalChoice = int(thal_combobox.get())
    except:
        messagebox.showerror("Error", "Please select thal!")
        return
    try:
        tresChoice = int(trestbps.get())
        cholChoice = int(chol.get())
        thalachChoice = int(thalach.get())
        oldpeakChoice = int(oldpeak.get())
    except:
        messagebox.showerror("Error", "Missing Data!")
        return

    print("A - age", A)
    print("B - gender", genChoice)
    print("A - age", fbsChoice)
    print("A - age", exangChoice)
    print("A - age", cpChoice)
    print("A - age", resChoice)
    print("A - age", slopeChoice)
    print("A - age", caChoice)
    print("A - ", thalChoice)
    print("A - slope", tresChoice)
    print("A - ca", cholChoice)
    print("M - thal", thalachChoice)
    print("M - thal", oldpeakChoice)

    figureFrame1 = Figure(figsize=(5, 5), dpi=100)
    a = figureFrame1.add_subplot(111)
    a.plot(['Sex', 'fbs', 'exang'], [genChoice, fbsChoice, exangChoice])
    canvas1 = FigureCanvasTkAgg(figureFrame1)
    canvas1.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand=True)
    canvas1._tkcanvas.place(width=250, height=250, x=600, y=240)

    figureFrame2 = Figure(figsize=(5, 5), dpi=100)
    a = figureFrame2.add_subplot(111)
    a.plot(['age', 'trestbps', 'chol', 'thalach'], [A, tresChoice, cholChoice, thalachChoice])
    canvas2 = FigureCanvasTkAgg(figureFrame2)
    canvas2.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand=True)
    canvas2._tkcanvas.place(width=250, height=250, x=880, y=240)

    figureFrame3 = Figure(figsize=(5, 5), dpi=100)
    a = figureFrame3.add_subplot(111)
    a.plot(['oldpeak', 'resticg', 'cp'], [oldpeakChoice, resChoice, cpChoice])
    canvas3 = FigureCanvasTkAgg(figureFrame3)
    canvas3.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand=True)
    canvas3._tkcanvas.place(width=250, height=250, x=600, y=520)

    figureFrame4 = Figure(figsize=(5, 5), dpi=100)
    a = figureFrame4.add_subplot(111)
    a.plot([ 'slope', 'ca', 'thal'], [slopeChoice, caChoice, thalChoice])
    canvas4 = FigureCanvasTkAgg(figureFrame4)
    canvas4.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand=True)
    canvas4._tkcanvas.place(width=250, height=250, x=880, y=520)

    input_data = (A,
        genChoice,
        fbsChoice,
        exangChoice,
        cpChoice,
        resChoice,
        tresChoice,
        cholChoice,
        thalChoice,
        thalachChoice,
        oldpeakChoice,
        slopeChoice,
        caChoice
    )

    input_array = np.asanyarray(input_data)

    input_reshape = input_array.reshape(1, -1)

    prediction = model.predict(input_reshape)
    print(prediction[0])

    if(prediction[0] == 0):
        print("The person does not have a heart disease")
        report.config(text = f"Report: {0}", fg="#8dc63f")
        report1.config(text=f"{name}, you do not have a heart disease")
    else:
        print("The person has a heart disease")
        report.config(text = f"Report: {1}", fg="#ed1c24")
        report1.config(text=f"{name}, you have a heart disease")


analysis_button = Button(root, text="Analysis", font="arial 10 bold", cursor="hand2", bg=background, command=handleAnalysis).place(x=1130, y=240)

info_button = Button(root, text="Info", font="arial 10 bold", cursor="hand2", background=background, command=Info).place(x=10, y =240)

save_button = Button(root, text="Save", font="arial 10 bold", cursor="hand2", background=background).place(x=1370, y =250)

isSmoking = True
choice = "smoking"

smoking_icon = PhotoImage(file="images/images1.png")
non_smoking_icon = PhotoImage(file="images/images2.png")

def smokingChoice():
    global isSmoking
    global choice
    if isSmoking:
        choice ="non_smoking"
        mode.config(image=non_smoking_icon, activebackground="white")
        isSmoking = False
    else: 
        choice = "smoking"
        mode.config(image=smoking_icon, activebackground="white")
        isSmoking = True
    print(choice)



smoking_icon = smoking_icon.subsample(6)
non_smoking_icon = non_smoking_icon.subsample(4)

mode = Button(root, image=smoking_icon, bg="#dbe0e3", bd=0, cursor="hand2", command=smokingChoice)
mode.place(x=350, y =495)

logout_button = Button(root, text="logout", bg="red", cursor="hand2", command=logout)
logout_button.place(x=1390, y=60)


root.mainloop()