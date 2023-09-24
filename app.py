import streamlit as st
import numpy as np

import pandas as pd

df = pd.read_pickle("df.pkl")
pipe = pd.read_pickle("pipe.pkl")

# pipe = pickle.load(open('pipe.pkl','rb'))
# df = pickle.load(open('df.pkl','rb'))

st.set_page_config(page_title="Laptop Prediction", page_icon="💻")

st.title("Laptop Price Predictor :money_with_wings:")

company = st.selectbox('Brand',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
ram = st.selectbox('RAM(GB)',[2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight(Kg)')
touchscreen = st.selectbox('Touchscreen',['Yes', 'No'])
ips = st.selectbox('IPS',['Yes', 'No'])
fullHD = st.selectbox('Full HD',['Yes', 'No'])
screen_size = st.number_input('Screen Size(In)',min_value=1.00)
resolution = st.selectbox('Screen Resolution',['1920x1080','1920x1200','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
clockSpeed = st.number_input('Clock Speed(Ghz)')
processor = st.selectbox('Processor',df['Processor'].unique())
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())
os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    touchscreen = 1 if (touchscreen == 'Yes') else 0
    ips = 1 if (ips == 'Yes') else 0
    fullHD = 1 if (fullHD == 'Yes') else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    
    query = np.array([company,type,ram,weight,touchscreen,ips,fullHD,ppi,clockSpeed,processor,ssd,hdd,gpu,os], dtype=object)
    query = query.reshape(1,14)
    
    # print(query)
    
    st.title("The predicted price of this configuration is Rs. " + str("{:.2f}".format(float(np.exp(pipe.predict(query)[0])))))
