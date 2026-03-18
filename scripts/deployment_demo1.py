import streamlit as st
import pandas as pd

st.write('## Deployment Project')

salary = st.text_input('How much is your salary?')

if(salary): 
    st.write(f'Your Salary is: {salary}')


df = pd.read_csv('coefficients.csv')

st.write(df)
