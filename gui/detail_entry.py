from tkinter import Frame, Label, Entry, IntVar, StringVar, Radiobutton
from tkinter.ttk import Combobox

def create_detail_entry(root):
    Detail_entry = Frame(root, width=650, height=260, bg="#dbe0e3")
    Detail_entry.place(x=30, y=450)

    Label(Detail_entry, text="sex:", font="arial 15", bg="#62a7ff", fg="#fefbfb").place(x=10, y=10)
    gen = IntVar()
    Radiobutton(Detail_entry, text="Male", variable=gen, value=1).place(x=60, y=14)
    Radiobutton(Detail_entry, text="Female", variable=gen, value=2).place(x=123, y=14)
