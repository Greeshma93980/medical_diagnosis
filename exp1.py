import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from sklearn.metrics import classification_report,confusion_matrix 
import numpy as np
train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
train=train_datagen.flow_from_directory(r"archive (9)\chest_xray\chest_xray\train",target_size=(150,150),batch_size=32,class_mode='binary')        #importing dtaset,converting to 150,150 pixesls,at one time take 32 pics,
test=test_datagen.flow_from_directory(r"archive (9)\chest_xray\chest_xray\test",target_size=(150,150),batch_size=32,class_mode='binary') 
#32 filters of mtrix size 3*3 ,1 img is conerted to 32 nodes
#input_shape=>150*150 image with3 colors RGB
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3), name="conv_1"),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu',),  # <--- use this in Grad-CAM
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=test, epochs=5)

# Save model
model.save("pneumonia_model.h5")
