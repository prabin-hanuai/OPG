#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import cv2
import numpy
from ultralytics import YOLO




def about():
    st.write('### Under Development....')

def dev():
    st.write('### Under Development....')
    


def main():
    st.title("OPG Anamoly detection ")

    activities = ["Home", "About", "Devloper"]
    choice = st.sidebar.selectbox("Pick something ", activities)

    model = YOLO('best (1).pt')

    
    if choice == "Home":
        
        st.write("Go to the About section from the sidebar to learn more about it.")
        
        # upload
        image_file = st.file_uploader("Upload image", type=['png'])
        
        if image_file is not None:
            
            #img = cv2.imread(image_file)
            img = cv2.imdecode(numpy.fromstring(image_file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            imgshow = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(imgshow)
            
            if st.button("Process"):
                result1 = model(imgshow)
                result1[0].save(boxes = False)
                annotated_image = result1[0].plot()
                st.image(annotated_image)

                    
                    

    elif choice == "About":
        about()
        
    elif choice == "Devloper":
        dev()



if __name__ == "__main__":
    main()
