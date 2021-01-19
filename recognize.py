import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import os
import numpy as np

#load the trained model to classify sign
from tensorflow.keras.models import load_model
model = load_model('trafficsign_classifier.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

cur_path = os.getcwd()
path = os.path.join(cur_path,'Meta')     
    
#initialise GUI
top=tk.Tk()
top.geometry('500x400')
top.title('Traffic Sign Classifier')
top.configure(background='#CDCDCD')

signname=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
prob=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
inputsign = Label(top)
resultsign = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image=image/255

    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    predictions = model.predict(image)
    probabilityValue =np.amax(predictions)
    print("Sign       : " + sign)
    print("Probability: " + str(round(probabilityValue*100,2)) + "%")
    signname.configure(foreground='#011638', text="Sign: "+sign)
    prob.configure(foreground='#011638', text="Probability: "+str(round(probabilityValue*100,2)) + "%")
    signimg=Image.open(path+'\\'+ str(pred) + ".png")
    signimg = signimg.resize((150, 150), Image.ANTIALIAS) 

    img=ImageTk.PhotoImage(signimg)
    resultsign.configure(image=img)
    resultsign.image=img

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.75, rely=0.9, anchor = CENTER)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded = uploaded.resize((150, 150), Image.ANTIALIAS) 

        im=ImageTk.PhotoImage(uploaded)
        
        inputsign.configure(image=im)

        inputsign.image=im
        signname.configure(text='')
        show_classify_button(file_path)
    except:
        pass

heading = Label(top, text="Traffic Sign Recognizer",pady=15, font=('arial',15,'bold'))
heading.configure(background='#CDCDCD',foreground='#011638')
heading.place(relx=0.5, rely=0.1 , anchor = CENTER)

upload_b=Button(top,text="Upload an Image",command=upload_image,padx=10,pady=5)
upload_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload_b.place(relx=0.25, rely=0.9, anchor = CENTER)

inputsign.place(relx=0.25, rely=0.6 , anchor = CENTER)
resultsign.place(relx=0.75, rely=0.6, anchor = CENTER)
signname.place(relx=0.5, rely=0.2, anchor = CENTER)
prob.place(relx=0.5, rely=0.3, anchor = CENTER)

top.mainloop()