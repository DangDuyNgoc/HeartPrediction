from tkinter import *

info_window = Tk()
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
Label(info_window, text="thal - 0 = normal; 1 = fixed defect; 2 reversible defect", font="arial 12").place(x=20, y=460)


info_window.mainloop()