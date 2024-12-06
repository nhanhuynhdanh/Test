import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Đang cài đặt {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Đã cài đặt xong {package}")

# Cài đặt các package cần thiết
required_packages = ['streamlit', 'joblib', 'numpy', 'scipy', 'regex', 'underthesea', 'pyvi']
for package in required_packages:
    install_package(package)


import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
import regex
from underthesea import word_tokenize, sent_tokenize
from pyvi import ViTokenizer
import re
import pickle
import pandas as pd


def load_models():
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('ExtraTreesClassifier_model_v1.pkl')
        with open('label_encoder.pkl', 'rb') as f:
            loaded_le = pickle.load(f)
            
        # In ra các classes của LabelEncoder
        print("Các nhãn trong LabelEncoder:", loaded_le.classes_)
        # Hoặc hiển thị trên Streamlit
        st.write("Các nhãn trong LabelEncoder:", loaded_le.classes_)
        
        return vectorizer, model, loaded_le
    except Exception as e:
        st.error(f"Lỗi khi tải models: {e}")
        return None, None, None
    
    
    
vectorizer, model, loaded_le = load_models()