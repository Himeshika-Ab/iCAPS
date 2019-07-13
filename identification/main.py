import tkinter as tk
from tkinter import filedialog
from tkinter import *
from identification import loadInput
from identification import processEcg

def onClick():
    print("clicked")
    filepath=filedialog.askopenfilename()
    # load raw ECG signal
    signal, mdata = loadInput.load_txt(filepath)
    # process it and plot
    out = processEcg.ecg(signal=signal, sampling_rate=1000., show=True)


root = tk.Tk()
root.title("iCAPS")

photo = PhotoImage(file="logo.png")
label = Label(root,image=photo)
label.pack()
button = tk.Button(root,text="Input file",fg="black",font="Calibri 15",command=onClick)
button.pack()

root.mainloop()



