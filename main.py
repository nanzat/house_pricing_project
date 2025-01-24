import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import pickle

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

st.markdown('''
    ### House Prices - Advanced Regression Techniques
    ###### :gray[Predict sales prices and practice feature engineering, RFs, and gradient boosting]
    ''')

st.link_button("View competition", "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview")

st.logo('./data/house/icon.jpg', icon_image='./data/house/images.jpg', size='large')

uploaded_file = st.sidebar.file_uploader(
    "Choose a data", type='csv'
)

if 'show_text' not in st.session_state:
    st.session_state.show_text = False

if st.sidebar.button("Our team"):
    st.session_state.show_text = not st.session_state.show_text

if st.session_state.show_text:
    st.sidebar.write('Anatoly')
    st.sidebar.write('Nanzat')
    st.sidebar.write('Ziyarat')

cdrop = ['YrSold', 'MoSold', '3SsnPorch', 'BsmtFinType2', 'LowQualFinSF', 'BsmtHalfBath', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id', 'SalePrice']
drops = ColumnTransformer(
    transformers=[
        ('drop', 'drop', cdrop)
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with open('./data/house/pipeline.pkl', 'rb') as file:
        model = pickle.load(file)

    itog = pd.DataFrame({'Id': df['Id'], 'SalePrice': np.exp(model.predict(df))})

    st.download_button(label="Download predictions",
                       data=convert_df(itog),
                       file_name='predictions.csv',
                       mime='text/csv')