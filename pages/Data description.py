import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import norm
df = pd.read_csv('data/house/train.csv')

st.markdown('''
    ## House Pricing dataset description
''')
st.dataframe(df.head())

st.markdown("""
 #### <- In the sidebar you can select the plots to be displayed
""")
st.sidebar.write("""
## Available plots
""")
if st.sidebar.checkbox("Target distribution", key="tips"):
    fig, ax = plt.subplots()
    (mu, sigma) = norm.fit(df['SalePrice'])
    sns.distplot(df['SalePrice'], kde=True, hist=True, fit=norm)
    plt.title('SalePrice distribution vs Normal Distribution', fontsize=13)
    plt.xlabel("House's sale Price in $", fontsize=12)
    plt.legend(['actual price dist', 'Normal dist ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download distplot",
            data=file,
            file_name="Tips_by_time.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Logarithm target distribution", key="logs"):
    fig, ax = plt.subplots()
    log_dist = np.log(df['SalePrice'])
    (mu, sigma) = norm.fit(log_dist)
    sns.distplot(log_dist, kde=True, hist=True, fit=norm)
    plt.title('Log(SalePrice) distribution vs Normal Distribution', fontsize=13)
    plt.xlabel("House's sale Price in log($)", fontsize=12)
    plt.legend(['log(price) dist', 'Normal dist ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download log dist",
            data=file,
            file_name="Tips_by_time.png",
            mime="image/png",
            key='fasfa'
        )

if st.sidebar.checkbox("Correlation Matrix", key="corr"):
    st.image('data/house/output2.png')
    with open("data/house/output2.png", "rb") as file:
        btn = st.download_button(
            label="Download corr matrix",
            data=file,
            file_name="corr.png",
            mime="image/png",
            key='matr'
        )

if st.sidebar.checkbox("Nan values", key="nv"):
    st.image('data/house/output8.png')
    with open("data/house/output8.png", "rb") as file:
        btn = st.download_button(
            label="Download barplot",
            data=file,
            file_name="bar.png",
            mime="image/png",
            key='v'
        )

if st.sidebar.checkbox("Pipeline", key="pp"):
    st.image('data/house/output7.png')
    with open("data/house/output7.png", "rb") as file:
        btn = st.download_button(
            label="Download pipeline",
            data=file,
            file_name="pipe.png",
            mime="image/png",
            key='pipe'
        )

if st.sidebar.checkbox("Real vs Predicted", key="vs"):
    st.image('data/house/output4.png')
    with open("data/house/output4.png", "rb") as file:
        btn = st.download_button(
            label="Download scatterplot",
            data=file,
            file_name="pipe.png",
            mime="image/png",
            key='rvsp'
        )

if st.sidebar.checkbox("Residuals", key="rs"):
    st.image('data/house/output5.png')
    with open("data/house/output5.png", "rb") as file:
        btn = st.download_button(
            label="Download histplot",
            data=file,
            file_name="pipe.png",
            mime="image/png",
            key='hist'
        )

if st.sidebar.checkbox("Training line", key="tl"):
    st.image('data/house/output6.png')
    with open("data/house/output6.png", "rb") as file:
        btn = st.download_button(
            label="Download lineplot",
            data=file,
            file_name="pipe.png",
            mime="image/png",
            key='lp'
        )

