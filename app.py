from zipfile import Path
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st 
import pickle
import skimage.io
import skimage.transform
import skimage.color
import skimage.feature
import scipy
import cv2


def pipeline(image,model,scaler):
    #resize the image into 80*80 as we have trained our all images at 80 by 80only
    # width,height = img.size
    # im1 = img.crop((0, 0, 0, 0))
    # newsize = (80, 80)
    # resized_image = np.array(im1.resize(newsize))
    img = skimage.io.imread(image)
    st.image(img)
    resized_image = skimage.transform.resize(img,(80,80))
    scaled_resized_image = 255*resized_image
    final_img = scaled_resized_image.astype(np.uint8)

    #convertion to gray
    gray_img = skimage.color.rgb2gray(final_img)

    #conversion to hog 
    hog_img = skimage.feature.hog(final_img)
    hog_img_reshaped = hog_img.reshape(1,-1)
    scaled_img = scaler.transform(hog_img_reshaped)

    value = model.decision_function(scaled_img)
    classes = model.classes_

    z = scipy.stats.zscore(value[0])
    prob_value = scipy.special.softmax(z)

    top_5 = prob_value.argsort()[::-1][:5]
    top_5_values = prob_value[top_5]

    matched_labels = classes[top_5]
    st.subheader("Probability of that image belongs to the below respective class is: ")
    for i in range(5):
        st.write(matched_labels[i],np.round(top_5_values[i],3))
    st.success(matched_labels[0])

model = pickle.load(open("Image_Classification.pickle",'rb'))
scaler = pickle.load(open("Scaler.pickle","rb"))

st.subheader("Image Classification")
image = st.file_uploader(label="Upload your Image here",type=['png','jpg','jpeg','bmp'])
if image is not None:
    pipeline(image,model,scaler)
    st.balloons()

st.info("Please Upload an Image")
st.caption("Made with ❤️ by TejaChavva")



