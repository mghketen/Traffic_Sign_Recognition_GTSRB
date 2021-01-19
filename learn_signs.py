import os
import numpy as np 
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

X = []
Y = []
class_names = ['Speed limit (20km/h)', 'Speed limit (30km/h)',      
           'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)',     
           'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',     
           'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry', 'General caution',     
           'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',  
           'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
           'Wild animals crossing', 'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',      
           'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons' ]
classes = 43
cur_path = os.getcwd()
print("Loading Images and Labels...")
#Retrieving the images and their labels 
for i in range(classes):
    print("Loading Images and Labels of %s" %(class_names[i]))
    path = os.path.join(cur_path,'Train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((32,32))
            image = np.array(image)
            image=image/255

            X.append(image)
            Y.append(i)
        except:
            print("Error loading image")

#Converting lists into numpy arrays
X = np.array(X)
Y = np.array(Y)
print("Total no. of Classes = %d" %classes)
print("Loading completed")
print("No. of Images = %d, No. of Labels= %d" %(X.shape[0], Y.shape[0]))
#Splitting training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Spliting images for training and testing")
print("No. of Training Images = %d, No. of Training Labels= %d" %(X_train.shape[0], Y_train.shape[0]))
print("No. of Testing Images = %d, No. of Testing Labels= %d" %(X_test.shape[0], Y_test.shape[0]))

del X
del Y

#Converting the labels into one hot encoding
Y_train = to_categorical(Y_train, 43)
Y_test = to_categorical(Y_test, 43)

#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
mc = ModelCheckpoint('model_best.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)
history = model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_test, Y_test),callbacks=[early_stop, mc])
best_model= load_model('model_best.h5')
print("--> Saving model as .h5")
best_model.save("trafficsign_classifier.h5")

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

Y_val = pd.read_csv('Test.csv')

labels = Y_val["ClassId"].values
imgs = Y_val["Path"].values

X_val=[]

for img in imgs:
    image = Image.open(img)
    image = image.resize((32,32))
    image = np.array(image)
    image=image/255
    X_val.append(image)

X_val=np.array(X_val)

pred = best_model.predict_classes(X_val)

#Accuracy with the test data
print("Accuracy Score = %f" %((accuracy_score(labels, pred))*100)+"%")