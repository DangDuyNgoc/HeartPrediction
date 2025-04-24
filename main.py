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

# Load the Random Forest CLassifier model

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure
import numpy as np

from trainingData import *
from db_connection import connect_db

filename = 'heart-disease-prediction-model.pkl'
model = pickle.load(open(filename, 'rb'))

background = "#f0ddd5"
framebg = "#62a7ff"
framefg= "#fefbfb"

# Connect to the database
try:
    db_connection = connect_db()
    print("Database connected successfully!")
except Exception as e:
    print(f"Error connecting to database: {e}")
    db_connection = None

root = Tk()
root.title("Heart Attack Prediction")
root.geometry("1450x800+40+20") #chiều rộng là 1450 pixel và chiều cao là 730 pixel 
 # 40 vị trí từ phía bên trái của màn hình và 20 chỉ vị trí từ phía trên .
root.resizable(False, False) #không thể kéo căn chỉnh kích thước của cửa sổ bằng chuột.
root.config(bg=background)

image_icon = PhotoImage(file='images/icon.png')
root.iconphoto(False, image_icon)



logo = PhotoImage(file='images/header.png')
header = Label(image=logo, bg=background)
header.place(x=0, y=0)

def display_info(language):
    info_window = Toplevel(root)
    info_window.title("info")
    info_window.geometry("700x600+100+100")

    if language == 'English':
        info_text = [
            "age - age in years",
            "sex - sex (1 = male; 0 = female)",
            "cp - chest pain type(0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)",
            "trestbps - resting blood pressure (in mm Hg on admission to the hospital)",
            "chol - serum cholesterol in mg/dl",
            "fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
            "restecg - resting electrocardiographic results (0 = normal; 1 = ST-T wave abnormality; 2 = showing probable or definite left ventricular hypertrophy)",
            "thalach - maximum heart rate achieved",
            "exang - exercise-induced angina (1 = yes; 0 = no)",
            "oldpeak - ST depression induced by exercise relative to rest",
            "slope - the slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)",
            "ca - number of major vessels (0-3) colored by fluoroscopy",
            "thal - Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)"
        ]
    else:  # Vietnamese
        info_text = [
            "age - tuổi tính bằng năm",
            "sex - giới tính (1 = nam; 0 = nữ)",
            "cp - loại đau ngực (0 = đau thắt ngực điển hình; 1 = đau thắt ngực không điển hình; 2 = đau không do mạch vành; 3 = không có triệu chứng)",
            "trestbps - huyết áp nghỉ ngơi (mm Hg khi nhập viện)",
            "chol - cholesterol huyết thanh (mg/dl)",
            "fbs - đường huyết đói > 120 mg/dl (1 = đúng; 0 = sai)",
            "restecg - kết quả điện tâm đồ nghỉ (0 = bình thường; 1 = ST-T bất thường; 2 = phì đại thất trái rõ rệt hoặc chắc chắn)",
            "thalach - nhịp tim tối đa đạt được",
            "exang - đau thắt ngực do tập thể dục (1 = có; 0 = không)",
            "oldpeak - mức độ trầm cảm ST do tập thể dục so với lúc nghỉ",
            "slope - độ dốc của đoạn ST khi tập thể dục (0 = lên dốc; 1 = phẳng; 2 = xuống dốc)",
            "ca - số lượng mạch chính (0-3) được làm nổi bật bởi fluoroscopy",
            "thal - Thalassemia (0 = bình thường; 1 = khiếm khuyết cố định; 2 = khiếm khuyết hồi phục)"
        ]

    Label(info_window, text="Information related to dataset", font="robot 15 bold").pack(padx=20, pady=20)
    for idx, text in enumerate(info_text):
        Label(info_window, text=text, font="arial 12").place(x=20, y=100 + 30 * idx)

    info_window.mainloop()

def Info():
    info_window = Toplevel(root)
    info_window.title("Info")
    info_window.geometry("400x200+100+100")

    Label(info_window, text="Choose Language", font="robot 15 bold").pack(padx=20, pady=20)

    Button(info_window, text="English", font="arial 12", command=lambda: display_info('English')).pack(pady=10)
    Button(info_window, text="Vietnamese", font="arial 12", command=lambda: display_info('Vietnamese')).pack(pady=10)

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
Heading_entry.place(x=600, y = 20)

Label(Heading_entry, text="Registration No.", font="arial 15", bg= "#df2d4b", fg= framefg).place(x=30, y =0)
Label(Heading_entry, text="Date", font="arial 15", bg= "#df2d4b", fg= framefg).place(x= 430, y =0)
Label(Heading_entry, text="Patient Name", font="arial 15", bg= "#df2d4b", fg= framefg).place(x= 30, y =90)
Label(Heading_entry, text="Birth Year", font="arial 15", bg= "#df2d4b", fg= framefg).place(x= 430, y = 90)

Entry_image = PhotoImage(file='images/Rounded Rectangle 1.png')
Entry_image2 = PhotoImage(file='images/Rounded Rectangle 2.png')
Label(Heading_entry, image=Entry_image, bg= "#df2d4b").place(x= 20, y =30)
Label(Heading_entry, image=Entry_image, bg= "#df2d4b").place(x= 430, y =30)

Label(Heading_entry, image=Entry_image2, bg= "#df2d4b").place(x= 20, y =120)
Label(Heading_entry, image=Entry_image2, bg= "#df2d4b").place(x= 430, y =120)

Registration = IntVar()
reg_entry = Entry(Heading_entry, textvariable=Registration, width=20, font= "arial 15", bg= "#0e5363", fg= "white", bd= 0)
reg_entry.place(x=30, y =45)

Date = StringVar()
today = date.today()
d1 = today.strftime("%Y/%m/%d")
date_entry = Entry(Heading_entry, textvariable=Date, width=15, font= "arial 15", bg= "#0e5363", fg= "white", bd= 0)
date_entry.place(x = 440, y = 45)
Date.set(d1)

Name = StringVar()
name_entry = Entry(Heading_entry, textvariable=Name, width=20, font= "arial 15", bg= "#ededed", fg= "#222222", bd= 0)
name_entry.place(x = 30, y = 135)

DayOfYear = IntVar()
year_entry = Entry(Heading_entry, textvariable=DayOfYear, width=20, font= "arial 15", bg= "#ededed", fg= "#222222", bd= 0)
year_entry.place(x = 440, y = 135)

Detail_entry = Frame(root, width = 650, height= 260, bg = "#dbe0e3")
Detail_entry.place(x = 30, y = 450)

Label(Detail_entry, text = "sex:", font = "arial 15", bg = framebg, fg = framefg).place(x = 10, y =10)
Label(Detail_entry, text = "fbs:", font = "arial 15", bg = framebg, fg = framefg).place(x = 210, y =10)
Label(Detail_entry, text = "exang:", font = "arial 15", bg = framebg, fg = framefg).place(x = 388, y =10)

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
RMale.place(x = 60, y = 14)
RFe.place(x = 123, y = 14)

fbs = IntVar() # đường huyết sau một thời gian nhịn ăn
RTrue = Radiobutton(Detail_entry, text= "True", variable=fbs, value= 1, command=fbsSelection)
RFalse = Radiobutton(Detail_entry, text= "False", variable=fbs, value= 2, command=fbsSelection)
RTrue.place(x = 258, y =14)
RFalse.place(x = 312, y = 14)

exang = IntVar() # tập thể dục gây đau thắt ngực Exercise induced angina
R1 = Radiobutton(Detail_entry, text= "Yes", variable=exang, value= 1, command=exangSelection)
R2 = Radiobutton(Detail_entry, text= "No", variable=exang, value= 2, command=exangSelection)
R1.place(x = 470, y =14)
R2.place(x = 520, y = 14)

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
ca_combobox = Combobox(Detail_entry, values= ['0', '1', '2', '3'], font="arial 12", state="r", width=14)
thal_combobox = Combobox(Detail_entry, values= ['0', '1', '2'], font="arial 12", state="r", width=14)

cp_combobox.place(x=44, y=52)
restecg_combobox.place(x=100, y=93)
slope_combobox.place(x=78, y=133)
ca_combobox.place(x=44, y=172)
thal_combobox.place(x=58, y=212)

Label(Detail_entry, text="trestbps", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=260, y = 90)
Label(Detail_entry, text="chol", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=260, y = 130)
Label(Detail_entry, text="thalach", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=260, y = 170)
Label(Detail_entry, text="odlpeak", font="arial 13", width=7, bg=framebg, fg=framefg).place(x=260, y = 210)

trestbps = StringVar()
chol = StringVar()
thalach = StringVar()
oldpeak = StringVar()

trestbps_entry = Entry(Detail_entry, textvariable=trestbps, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)
chol_entry = Entry(Detail_entry, textvariable=chol, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)
thalach_entry = Entry(Detail_entry, textvariable=thalach, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)
oldpeak_entry = Entry(Detail_entry, textvariable=oldpeak, width= 10, font= "arial 15", bg = "#ededed", fg="#222222", bd=0)

trestbps_entry.place(x=360, y=90)
chol_entry.place(x=360, y=130)
thalach_entry.place(x=360, y=170)
oldpeak_entry.place(x=360, y=210)

# Report
report_image = PhotoImage(file='images/Report.png')
report_bg = Label(image=report_image, bg=background)
report_bg.place(x=1180, y=320)

report = Label(root, font="arial 20 bold", bg="white", fg="#8dc63f")
report.place(x=1225, y=550)

report1 = Label(root, font="arial 10 bold", bd=0)
report1.place(x=1180, y=610)

# graph
graph_image = PhotoImage(file='images/graph.png')
Label(image=graph_image).place(x=600, y=270)
Label(image=graph_image).place(x=860, y=270)
Label(image=graph_image).place(x=600, y=500)
Label(image=graph_image).place(x=860, y=500)

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
        oldpeakChoice = float(oldpeak.get())
    except:
        messagebox.showerror("Error", "Missing Data!")
        return

    print("A - age", A)
    print("gender: ", genChoice)
    print("fbsChoice: ", fbsChoice)
    print("exangChoice: ", exangChoice)
    print("cpChoice: ", cpChoice)
    print("resChoice: ", resChoice)
    print("slopeChoice: ", slopeChoice)
    print("caChoice: ", caChoice)
    print("thalChoice: ", thalChoice)
    print("tresChoice: ", tresChoice)
    print("cholChoice: ", cholChoice)
    print("thalachChoice: ", thalachChoice)
    print("oldpeakChoice: ", oldpeakChoice)

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

    # knn_prediction = clf_1.predict(input_reshape)

    if(prediction[0] == 0):
        print("The person does not have a heart disease")
        report.config(text = f"Report: {0}", fg="#8dc63f")
        report1.config(text=f"{name}, you dont have heart disease")
    else:
        print("The person has a heart disease")
        report.config(text = f"Report: {1}", fg="#ed1c24")
        report1.config(text=f"{name}, you have heart disease")

def save_patient_data(registration_no, patient_name, birth_year):
    if db_connection: 
        try:
            cursor = db_connection.cursor()
            query = """
                INSERT INTO patients (registration_no, patient_name, birth_year)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (registration_no, patient_name, birth_year))
            db_connection.commit()
            print("Patient data saved successfully!")
            return cursor.lastrowid  # Return the ID of the newly inserted patient
        except Exception as e:
            print(f"Error saving patient data: {e}")
        finally:
            cursor.close()
    else:
        print("No database connection available.")

def save_prediction_input(patient_id, date, sex, cp, fbs, restecg, trestbps, chol, thalach, exang, oldpeak, slope, ca, thal):
    if db_connection:
        try:
            cursor = db_connection.cursor()
            query = """
                INSERT INTO prediction_inputs (patient_id, date, sex, cp, fbs, restecg, trestbps, chol, thalach, exang, oldpeak, slope, ca, thal) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (patient_id, date, sex, cp, fbs, restecg, trestbps, chol, thalach, exang, oldpeak, slope, ca, thal))
            db_connection.commit()
            print("Prediction input saved successfully!")
            return cursor.lastrowid  # Return the ID of the newly inserted prediction input
        except Exception as e:
            print(f"Error saving prediction input: {e}")
        finally:
            cursor.close()
    else:
        print("No database connection available.")

def save_prediction_result(input_id, result):
    if db_connection:
        try:
            cursor = db_connection.cursor()
            query = """
                INSERT INTO prediction_results (input_id, result) 
                VALUES (%s, %s)
            """
            cursor.execute(query, (input_id, result))
            db_connection.commit()
            print("Prediction result saved successfully!")
        except Exception as e:
            print(f"Error saving prediction result: {e}")
        finally:
            cursor.close()
    else:
        print("No database connection available.")

def Save():

    # get data of patients
    registration_no = Registration.get()
    patient_name = Name.get()
    birth_year = DayOfYear.get()

    # save patient data to database
    patient_id = save_patient_data(registration_no, patient_name, birth_year)
    if not patient_id:
        messagebox.showerror("Error", "Failed to save patient data.")
        return
    
    # get data of prediction inputs
    date = Date.get()
    try: 
        sex = genSelection()
        cp = cpSelection()
        trestbps_value = int(trestbps.get())
        chol_value = int(chol.get())
        thalach_value = int(thalach.get())
        fbs_value = fbsSelection()
        restecg_value = int(restecg_combobox.get())
        exang_value = exangSelection()
        oldpeak_value = float(oldpeak.get())
        slope_value = int(slopeSelection())
        ca_value = int(ca_combobox.get())
        thal_value = int(thal_combobox.get())
    except Exception as e:
        messagebox.showerror("Error", "Please fill in all fields correctly.")
        return
    
    # save prediction input to database
    input_id = save_prediction_input(patient_id, date, sex, cp, fbs_value, restecg_value, trestbps_value, chol_value, thalach_value, exang_value, oldpeak_value, slope_value, ca_value, thal_value)

    input_data = (birth_year, sex, cp, trestbps_value, chol_value, fbs_value, restecg_value, thalach_value, exang_value, oldpeak_value, slope_value, ca_value, thal_value)
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    result = int(prediction[0])

    # save prediction result to database
    save_prediction_result(input_id, result)

    # show the message box 
    messagebox.showinfo("Success", "Record Saved Successfully")

# analysis button
analysis_button = PhotoImage(file='images/Analysis.png')
analysis_button = analysis_button.subsample(2)
Button(root, image=analysis_button, bd=0, bg=background, cursor='hand2', command=handleAnalysis).place(x=1200, y=260)

# info button
info_button = PhotoImage(file='images/info.png')
Button(root, image=info_button, cursor="hand2", bd=0, background=background, command=Info).place(x=10, y =240)

# save button
save_button = PhotoImage(file='images/save.png')
Button(root, image=save_button, cursor="hand2", bd=0, background=background, command=Save).place(x=1320, y=260)

# logout button
logout_button = PhotoImage(file='images/logout_icon.png')
logout_button = logout_button.subsample(2)
Button(root, image=logout_button, bg=background, bd=0, cursor="hand2", command=logout).place(x=1370, y=220)

root.mainloop()