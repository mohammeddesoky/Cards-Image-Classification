import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import seaborn as sns
import os
import cv2
import random
import glob as gb
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import font
import PIL
from PIL import  Image,ImageTk
import customtkinter as ctk
print("Done")

model = keras.models.load_model("D:\\NN project\\best_model_72.keras")
def img_preprocessing(img):
    img = cv2.resize(img, (s, s))  
    img = img / 255.0
    img = np.array(img)
    return img
s=100
categorys = os.listdir("D:\\NN project\\archive\\train")
def browseFiles():
    
    
    filename = filedialog.askopenfilename(initialdir = "D:\\NN project\\archive",
                                          title = "Select a File",
                                          filetypes = (("JPG Images",
                                                        "*.jpg"),
                                                       ("all files",
                                                        "*.*")))
    
    print(filename)
    
    ## Read And Display The Image
    img=cv2.imread(filename)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    frame2=Frame(root)
    frame2.configure(background=bg,padx=50,pady=80)
    frame2.place(anchor="e",relx=0.97,rely=0.52)
    pred_result.set("")
    fig=Figure(figsize=(3,3))
    
    ax=fig.add_subplot(111)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    canvas=FigureCanvasTkAgg(fig,master=frame2)
    canvas.draw()
    canvas.get_tk_widget().pack(side=RIGHT)


    ## Generate Predictions
    image_processed=img_preprocessing(img)
    pred=model.predict(image_processed[np.newaxis, ...])
    m=np.argmax(pred)
    res=categorys[m]
    pred_result.set(res)
    print(res)
    # messagebox.showinfo("pre",pred[0][1])
    #messagebox.showinfo("Prediction Result",str(res))
    


    

    
def set_center():
    w = 900
    h = 750
    #root.state('zoomed')
    root.resizable(0,0)
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    x = int((screenwidth-w)/2)
    y = int((screenheight-h)/2)
    root.geometry(f'{w}x{h}+{x}+{y}')


# image_output=np.ones((50,50,3))

root=Tk()
root.title("Cards Classification - NN Project")

pred_result=StringVar()
pred_result.set("")
font=("Time New Romans",15)
btn_font=("Time New Romans",11,"bold")
bg="#caf0f8"
fg="#03045e"
set_center()
root.configure(background=bg)
## top frame
frame1 = Frame(root)
frame1.configure(background=bg)
frame1.place(anchor="center",relx=0.48,rely=0.07)

lbl_choose=Label(frame1,text="Explore for image to predict : ",fg=fg,bg=bg,font=font,width=25,padx=5,pady=2)
lbl_choose.grid(column=0,row=0,padx=10)

#btn_choose=Button(frame1,text="Browse Files",bg="white",fg=fg,padx=10,pady=3,font=btn_font,relief="ridge",command = browseFiles,borderwidth=5)
btn_choose=ctk.CTkButton(frame1,text="Browse Files",command=browseFiles)

btn_choose.grid(column=1,row=0)

## bottom right frame

# image_tk=cv2.imread("D:\\NN project\\archive\\valid\\ace of hearts\\2.jpg")
# # image_tk=cv2.cvtColor(image_tk, cv2.COLOR_BGR2RGB)
# image_tk = cv2.resize(image_tk, (250, 250))  # resize image (optional)
# image_tk = Image.fromarray(image_tk)
# image_tk = ImageTk.PhotoImage(image_tk)


## bottom left frame
frame3=Frame(root)
frame3.configure(background=bg,padx=250,pady=5)
frame3.place(anchor="w",relx=-0.1,rely=0.5)

lbl_pred=Label(frame3,text="Predicted Class :",fg=fg,bg=bg,font=font,padx=2,pady=10)
lbl_pred.grid(column=0,row=0)

lbl_res=Label(frame3,textvariable=pred_result,font=font,bg=bg,fg=fg,padx=2,pady=5)
lbl_res.grid(column=0,row=1,pady=7)

root.mainloop()

