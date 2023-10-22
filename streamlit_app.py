import streamlit as st
from PIL import Image
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.preprocessing import image as kimage
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

base_model = tf.keras.applications.EfficientNetB2(
    weights="imagenet", input_shape=(224, 224, 3), include_top=False
)


for layer in base_model.layers:
    layer.trainable = False
model = Sequential()
model.add(base_model)
# model.add(GaussianNoise(0.25))
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
# model.add(GaussianNoise(0.25))
model.add(Dropout(0.25))
model.add(Dense(4, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy", "AUC", "Precision", "Recall"],
)



def main():
    st.title("Cancer Identification")
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png"])

    if uploaded_file is not None:
        filepath = "best_model.h5"
        model.load_weights(filepath)
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
            st.image(image,width=300)  

        with col2:
            img = kimage.load_img(uploaded_file,target_size=(224,224))
            imag = kimage.img_to_array(img)
            imaga = np.expand_dims(imag,axis=0) 
            ypred = model.predict(imaga)
            a=np.argmax(ypred,-1)
            if a==0:
                op="Adenocarcinoma"
            elif a==1:
                op="large cell carcinoma"
            elif a==2:
                op="normal (void of cancer)"
            else:
                op="squamous cell carcinoma" 

            
            st.markdown(f'<h1 style="text-align: center">{op} ({ypred[0][a].item()*100:.1f}%)</h1>',unsafe_allow_html=True)

if __name__ == "__main__":
    main()