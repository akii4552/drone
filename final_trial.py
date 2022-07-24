# from logging import exception
from PIL import Image
import cv2
import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np



st.title('Drone Detection App')
st.sidebar.title('Drone Detection Sidebar')
st.sidebar.subheader('parameters')
# @st.cache()
detection_type=st.sidebar.selectbox('Choose the App mode',['About APP','Run on IMage','Run on Video','Go live'])
if detection_type=='About APP':
    st.markdown('This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode. ')
    st.markdown('''
    About the Author: \n
    Hey this is Akash Thakur from SJCEM.\n

    If you had fun with the App, Kindly Do like our Daisy and Share it with your friends.\n

    You Don't Necessarily need a Drone to run this app you can use an image from google.


    SAMPLE OUTPUT:\n
    '''    )

    st.video('https://youtu.be/gB9IMWCgWds')

elif detection_type=="Run on IMage":
    st.sidebar.markdown('-----')
    confidence=st.sidebar.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.6 )
    st.sidebar.markdown('-----')
    image=st.sidebar.file_uploader("UPLOAD an IMAGE",type=['jpg','png','jpeg'])
    if image is None:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        model.conf = 0.25
        # img_output=np.array(Image.open(image))
        # output=model(img_output)
        output=model('drone5.jpg')
        st.sidebar.text("original image")
        st.sidebar.image('drone5.jpg')
        plt.imshow(np.squeeze(output.render()))
       
        st.image(output.render())

    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        model.conf = 0.25
        image = Image.open(image)
        img_array = np.array(image) # if you want to pass it to OpenCV
        output1=model(Image.fromarray(img_array))
        plt.imshow(np.squeeze(output1.render()))
        st.image(output1.render())
        
elif detection_type=="Go live":
    st.sidebar.markdown('-----')
    confidence=st.sidebar.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.6 )
    st.sidebar.markdown('-----')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.conf = 0.25
    st.title("Going LIve")
    confirm=st.checkbox('Start the webcam')
    if confirm:
        st.write("Making connection to your webcam......Wait for a while")

    FRAME_WINDOW=st.image([])
    # cam = cv2.VideoCapture(0)
    source=0
    while confirm:
        
        try:
        
            cam = cv2.VideoCapture(source)
            ret,frame=cam.read()
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = model(frame)
            FRAME_WINDOW.image(np.squeeze(results.render()))
        except Exception as e:
            st.write("Unable to connect to your webcam kindly retry the App") 
            source=source+1;   
    else:
        
        st.write('Kindly select the checkbox to start your webcam')


# elif detection_type=="Run on Video":
#     st.sidebar.markdown('-----')
#     confidence=st.sidebar.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.6 )
#     st.sidebar.markdown('-----')
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
#     model.conf = 0.25
#     vidframe=
#     image=st.sidebar.file_uploader("UPLOAD an IMAGE",type=['jpg','png','jpeg'])
        
  
        
        




