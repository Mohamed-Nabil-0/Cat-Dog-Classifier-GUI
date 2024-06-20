# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:20:21 2021

@author: mohamed180353
"""

# We do not want to run predict_classes method every time we want to test our model.
# That’s why we need a graphical interface. Here we will build the GUI using Tkinter python.

# Now create a new directory, copy your model (“model1_catsVSdogs_10epoch.h5”) to this directory.

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy
import pyttsx3
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


def playSound(txt):
    # initialize Text-to-speech engine
    engine = pyttsx3.init()
    # convert this text to speech
    text = txt
    engine.setProperty("rate", 150)                     #defualt 200
    engine.say(text)
    
    # saving speech audio into a file
    # engine.save_to_file(text, "python.mp3")
    
    
    # play the speech
    engine.runAndWait()




model = load_model('model1_catsVSdogs_10epoch.h5')
#dictionary to label all traffic signs class.
classes = { 
    0:'its a cat',
    1:'its a dog',
    -1:'Unknown Image',
    
 
}
#initialise GUI
top=tk.Tk()
top.geometry('1400x650')
top.resizable(False, False)

top.title('Cats VS Dogs Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('Tahoma',18,'italic bold underline'))
sign_image = Label(top)

def classify(img_path):
    img =image.load_img(img_path , target_size=(128,128))

    x=image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x,axis=0)
    pred = np.argmax(model.predict(x)[0], axis=-1)

    if pred==1:
        preds="it's a DOG"
    elif pred==0:
        preds="it's a CAT"
    else:
        pred="Unkonwn image"


    playSound(preds)
    label.configure(foreground='blue', text=preds)       #foreground='#011638'
 
def show_classify_button(file_path):
    
    classify_b=Button(top,text="Classify Image",
                      command=lambda: classify(file_path),
                      padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Cats VS Dogs Classification",pady=20, font=('Tahoma',20,'italic bold underline'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()



copyWrite = tk.Text(top,background='#CDCDCD', borderwidth=0,font=('times', 17, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed By M.Nabil","", "", "superscript")
copyWrite.configure(state="disabled",fg="blue"  )
copyWrite.pack(side="left")
copyWrite.place(x=1100, y=600)


top.mainloop()













