import streamlit as st
import pandas as pd

st.write('## Demonstration of Deployment GMSC')

feature_cnt = st.text_input('How many features do we have?')

if(feature_cnt):
    st.write(f'We have {feature_cnt} features')

df = pd.read_csv(r'..\data\cs-test.csv')

st.write(df)
