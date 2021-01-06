from tkinter import *
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from ChordInference import ChordInference

root = Tk()
root.title("BeatIt")
#root.geometry('600x300')
FILENAME = 'blue.png'
canvas = Canvas(root,width=250, height=250)
canvas.pack()
tk_img = PhotoImage(file = FILENAME)
canvas.create_image(125, 125, image=tk_img)



def open_file():
    fileName = fd.askopenfilename()
    model = ChordInference()
    if fileName is not None:
        observations = model.extractPitch(fileName)
        result = model.generateChords(observations)
        model.generateOutputFile(result)



btnOpen = Button(root, text='Open', command=lambda: open_file(), activebackground = "#33B5E5")
#btnOpen.pack(side=TOP, pady=10)
quit_button_window = canvas.create_window(10, 10, anchor='nw', window=btnOpen)

root.mainloop()