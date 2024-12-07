import streamlit as st
from streamlit.components.v1 import html
import pickle
import os 
import seaborn as sns
import joblib
import warnings
from function import*
import pandas as pd

os.getcwd()

import nltk
import os

# Tạo thư mục tùy chỉnh cho dữ liệu NLTK
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Thêm thư mục vào đường dẫn tìm kiếm của NLTK
nltk.data.path.append(nltk_data_dir)

# Tải tài nguyên 'punkt' vào thư mục tùy chỉnh
nltk.download('punkt', download_dir=nltk_data_dir)



###################################
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore') # Bỏ qua tất cả cảnh báo
warnings.filterwarnings("ignore", category=UserWarning)

import random
import pandas as pd
import json
import regex
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from pyvi import ViTokenizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import requests
import re
import regex
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from pyvi.ViTokenizer import tokenize




#Load pre-trained models
@st.cache_resource
# def load_models():
#     try:
#         # Load tất cả models từ file đã train
#         vectorizer = joblib.load('tfidf_vectorizer_new_v1.pkl')
#         model = joblib.load('ExtraTreesClassifier_model_new.pkl')
        
#         # Load label encoder
#         with open('label_encoder_new.pkl', 'rb') as f:
#             loaded_le = pickle.load(f)
#             # print("Loaded classes:", loaded_le.classes_)
            
#         return vectorizer, model, loaded_le
        
#     except Exception as e:
#         st.error(f"Lỗi khi tải models: {e}")
#         return None, None, None


def load_models():
    try:
        # Load label encoder
        with open('label_encoder_new.pkl', 'rb') as f:
            loaded_le = pickle.load(f)
        
        # Map classes theo đúng thứ tự của model
        loaded_le.classes_ = np.array(['negative', 'neutral', 'positive'])
        
        # Load vectorizer và fit
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=0.02,
            max_df=0.85,
            sublinear_tf=True
        )
        
        df = pd.read_csv('df_clean_v1.csv')
        df['new_content'] = df['new_content'].fillna('')
        vectorizer.fit(df['new_content'])
        
        # Load model
        model = joblib.load('ExtraTreesClassifier_model_new.pkl')
        
        print("Model expects:", model.n_features_in_)
        print("Vectorizer vocabulary size:", len(vectorizer.vocabulary_))
        print("Classes mapping:", dict(enumerate(loaded_le.classes_)))
        
        return vectorizer, model, loaded_le
        
    except Exception as e:
        st.error(f"Lỗi khi tải models: {e}")
        return None, None, None




# Load models và hiển thị labels
vectorizer, model, loaded_le = load_models()
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = re.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        # CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
        # # TRANSLATE English -> Vietnamese
        # sentence = ' '.join(english_dict[word] if word in english_dict else word for word in sentence.split())
        # new_sentence = new_sentence+ sentence + '. '

    document = new_sentence

    document = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '',document)
    document = re.sub(r'[\r\n]+', ' ', document)
    document = re.sub('[^\w\s]', ' ', document)
    document = re.sub('[\s]{2,}', ' ', document)
    document = re.sub('^[\s]{1,}', '', document)
    document = re.sub('[\s]{1,}$', '', document)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~’'
    for char in punctuation:
        document = document.replace(char, ' ')

    return document

# Normalize unicode Vietnamese
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Unicode Vietnamese
def covert_unicode(txt):
    dicchar = loaddicchar()
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count, word_list


def remove_stopword(text, stopwords):
    """
    Loại bỏ stopwords khỏi văn bản.
    """
    # Kiểm tra nếu text là danh sách thì nối thành chuỗi
    if isinstance(text, list):
        text = ' '.join(text)

    # Loại bỏ stopwords
    document = ' '.join('' if word in stopwords else word for word in text.split())

    # Loại bỏ khoảng trắng dư thừa
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

neutral_words = ["chấp nhận được", "trung bình", "bình thường", "tạm ổn", "trung lập", "có thể"
                 "không nổi bật", "đủ ổn", "đủ tốt", "có thể chấp nhận", "bình thường",
                 "thường xuyên", "tương đối", "hợp lý", "tương tự",
                 "có thể sử dụng", "bình yên", "bình tĩnh", "không quá tệ", "trung hạng",
                 "có thể điểm cộng", "dễ chấp nhận", "không phải là vấn đề",
                 "không phản đối", "không quá đáng kể", "không gây bất ngờ", "không tạo ấn tượng", "có thể chấp nhận",
                 "không gây sốc", "tương đối tốt", "không thay đổi", "không quá phức tạp", "không đáng kể",
                 "chấp nhận", "có thể dễ dàng thích nghi", "không quá cầu kỳ", "không cần thiết", "không yêu cầu nhiều", "không gây hại",
                 "không có sự thay đổi đáng kể", "không rõ ràng", "không quá phê bình", "không đáng chú ý", "không đặc biệt",
                 "không quá phức tạp", "không gây phiền hà", "không đáng kể", "không gây kích thích"]

negative_words = [
    "kém", "tệ", "đau", "xấu", "bị","rè", "ồn",
    "buồn", "rối", "thô", "lâu", "sai", "hư", "dơ", "không có"
    "tối", "chán", "ít", "mờ", "mỏng", "vỡ", "hư hỏng",
    "lỏng lẻo", "khó", "cùi", "yếu", "mà", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp", "nhầm lẫn"
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp", "bị mở", "bị khui", "không đúng", "không đúng sản phẩm",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập", "bị bóc", "sai sản phẩm",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng", "giảm chất lượng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp"
]

positive_words = [
    "thích", "tốt", "xuất sắc","đúng", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn"
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh"
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng_lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền"
]

# List of words with negative meanings
negation_words = ["không", "nhưng", "tuy nhiên", "mặc dù", "chẳng", "mà", 'kém', 'giảm']

positive_emojis = [
    "😄", "😃", "😀", "😁", "😆",
    "😅", "🤣", "😂", "🙂", "🙃",
    "😉", "😊", "😇", "🥰", "😍",
    "🤩", "😘", "😗", "😚", "😙",
    "😋", "😛", "😜", "🤪", "😝",
    "🤗", "🤭", "🥳", "😌", "😎",
    "🤓", "🧐", "👍", "🤝", "🙌", "👏", "👋",
    "🤙", "✋", "🖐️", "👌", "🤞",
    "✌️", "🤟", "👈", "👉", "👆",
    "👇", "☝️"
]

# Count emojis positive and negative
negative_emojis = [
    "😞", "😔", "🙁", "☹️", "😕",
    "😢", "😭", "😖", "😣", "😩",
    "😠", "😡", "🤬", "😤", "😰",
    "😨", "😱", "😪", "😓", "🥺",
    "😒", "🙄", "😑", "😬", "😶",
    "🤯", "😳", "🤢", "🤮", "🤕",
    "🥴", "🤔", "😷", "🙅‍♂️", "🙅‍♀️",
    "🙆‍♂️", "🙆‍♀️", "🙇‍♂️", "🙇‍♀️", "🤦‍♂️",
    "🤦‍♀️", "🤷‍♂️", "🤷‍♀️", "🤢", "🤧",
    "🤨", "🤫", "👎", "👊", "✊", "🤛", "🤜",
    "🤚", "🖕"
]



# Load external files
def load_resources():
    files = {
        "emoji_dict": 'files/emojicon.txt',
        "teen_dict": 'files/teencode.txt',
        "wrong_lst": 'files/wrong-word.txt',
        "stopwords_lst": 'files/vietnamese-stopwords.txt',
    }
    resources = {}
    for key, path in files.items():
        with open(path, 'r', encoding='utf8') as file:
            if key.endswith('_dict'):
                resources[key] = {line.split('\t')[0]: line.split('\t')[1] for line in file.read().split('\n')}
            else:
                resources[key] = file.read().split('\n')
    return resources

resources = load_resources()

def predict_sentiment(text, vectorizer, model, loaded_le):
    try:
        # Tiền xử lý văn bản
        processed_text = process_text(text, emoji_dict, teen_dict, wrong_lst)
        processed_text = covert_unicode(processed_text)
        processed_text = remove_stopword(processed_text, stopwords_lst)
        
        # Vector hóa text
        text_features = vectorizer.transform([processed_text])
        
        # Đếm từ theo loại
        neutral_count = find_words(processed_text, neutral_words)[0]
        negative_count = find_words(processed_text, negative_words)[0]
        positive_count = find_words(processed_text, positive_words)[0]
        positive_emoji_count = find_words(text, positive_emojis)[0]
        negative_emoji_count = find_words(text, negative_emojis)[0]
        
        # Kết hợp features
        additional_features = np.array([
            neutral_count,
            negative_count,
            positive_count,
            positive_emoji_count,
            negative_emoji_count
        ]).reshape(1, -1)
        
        combined_features = hstack((text_features, additional_features))
        
        # Dự đoán xác suất cho từng class
        probas = model.predict_proba(combined_features)[0]
        print("Probabilities:", dict(zip(loaded_le.classes_, probas)))
        
        # Xác định sentiment dựa trên rules
        if negative_count > positive_count:
            sentiment = 'negative'
        elif positive_count > negative_count:
            sentiment = 'positive'
        elif neutral_count > 0:
            sentiment = 'neutral'
        else:
            # Nếu không có từ đặc trưng nào, dùng kết quả từ model
            prediction = model.predict(combined_features)[0]
            sentiment = loaded_le.classes_[prediction]
        
        print("Word counts:", {
            'neutral': neutral_count,
            'negative': negative_count,
            'positive': positive_count
        })
        print("Final sentiment:", sentiment)
        
        return {
            'sentiment': sentiment,
            'processed_text': processed_text,
            'features': {
                'neutral_words': neutral_count,
                'negative_words': negative_count,
                'positive_words': positive_count,
                'positive_emojis': positive_emoji_count,
                'negative_emojis': negative_emoji_count
            }
        }
    except Exception as e:
        print(f"Error details: {e}")
        return {'error': str(e)}






######################################
try:
    df_summary = pd.read_csv('df_summary.csv')
except Exception as e:
    st.error(f"Error loading data: {e}")

df_goc = df_summary[['id','ma_khach_hang', 'noi_dung_binh_luan', 'ngay_binh_luan', 
                    'gio_binh_luan', 'so_sao', 'ma_san_pham']]

# Sidebar menu
menu = ["Home","Bussiness Problem","Data Preprocessing","EDA","Modeling - Evaluation", "Comment Sentiment Analysis", "Product Sentiment Analysis"]

choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    # Title and logo
    logo_path = "image/logoTTTH.png"  # Update path if necessary
    st.image(logo_path, width=150)

    # Title with gradient effect
    st.markdown("""
    <h1 style="text-align: center; color: #ff6347; font-size: 50px; background: linear-gradient(to right, #ff7f50, #ff6347); -webkit-background-clip: text; color: transparent;">
        Trung Tâm Tin Học
    </h1>
    <h3 style="text-align: center; color: #4682B4;">
        Trường ĐH Khoa Học Tự Nhiên Tp. Hồ Chí Minh
    </h3>
    """, unsafe_allow_html=True)

    # Subtitle
    st.markdown("""
    <h2 style="color: #32CD32; text-align: center;">
        Đồ án tốt nghiệp Data Science
    </h2>
    <h3 style="text-align: center; font-style: italic; color: #4169E1;">
        Project 01: <b>Sentiment Analysis</b>
    </h3>
    """, unsafe_allow_html=True)

    # Display authors with styled text
    st.markdown("""
    <p style="text-align: center; color: #6A5ACD; font-size: 18px;">
        <b>Phan Minh Huệ</b><br>
        <b>Huỳnh Danh Nhân</b>
    </p>
    """, unsafe_allow_html=True)

    # Content outline with animated text color
    st.markdown("""
    <h4 style="color: #8B0000;">**Nội dung:**</h4>
    <ul style="font-size: 18px; line-height: 1.8; color: #483D8B;">
        <li><b>Business Understanding</b></li>
        <li><b>Data Preprocessing</b></li>
        <li><b>Exploratory Data Analysis (EDA)</b></li>
        <li><b>Modeling - Evaluation</b></li>
        <li><b>Comment Sentiment Analysis</b></li>
        <li><b>Product Sentiment Analysis</b></li>
    </ul>
    """, unsafe_allow_html=True)

    # Add a background effect for the page
    background_css = """
    <style>
    body {
        background: linear-gradient(to bottom, #f0f8ff, #ffffff);
    }
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)





elif choice == "Bussiness Problem":
    # Add a custom background using HTML and CSS
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            color: #333333;
        }
        .main {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #4b7bec;
            text-align: center;
            font-size: 32px;
        }
        h2 {
            color: #3742fa;
            font-size: 24px;
        }
        ul {
            font-size: 18px;
            color: #2f3542;
        }
        a {
            color: #1e90ff;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Add an image banner with a custom width
    image_path = "image/hasaki_banner_2.jpg"  # Update the path if necessary
    st.image(image_path, use_container_width =True, caption="Welcome to HASAKI")

    # Add a hyperlink in bold
    st.markdown(
        "**🌐 Tìm hiểu thêm tại:** [Hasaki.vn](https://hasaki.vn)",
        unsafe_allow_html=True,
    )
    
    # Title with styled font
    st.markdown('<h1>Bussiness Problem</h1>', unsafe_allow_html=True)

    # Introduction section
    st.markdown('<h2>Giới thiệu</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><b>Vấn đề Kinh Doanh:</b> HASAKI.VN, hệ thống cửa hàng mỹ phẩm và dịch vụ chăm sóc sắc đẹp hàng đầu, cần phân tích các đánh giá từ khách hàng để:</li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <ul>
            <li>Hiểu rõ hơn về ý kiến và nhu cầu của khách hàng.</li>
            <li>Nắm bắt phản hồi về sản phẩm và dịch vụ.</li>
            <li>Cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm, từ đó nâng cao sự hài lòng và lòng trung thành của khách hàng.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

elif choice == "Data Preprocessing":
    # Title
    st.title("Data Preprocessing")
    
    # Introduction
    st.write("### 1. Data Overview")
    st.write("""
    Bộ dữ liệu được cung cấp gồm các tệp:
    - **Danh_gia.csv**
    - **San_pham.csv**
    - **Khach_hang.csv**
    
    Sử dụng file **Danh_gia.csv** để xây dựng mô hình đánh giá.
    """)
    # Select the required columns


    # Display the table
    st.write("### Dữ liệu mẫu")
    st.dataframe(df_goc.head(5))  # Display first 5 rows
    # Data preprocessing steps
    st.write("### 2. Data Preprocessing Steps")
    st.markdown("""
    Các bước xử lí:
    - 📝 **Convert các từ tiếng Anh**  
    - 🔧 **Xử lí wrong word**  
    - ❌ **Xử lí stop word**  
    - 🎨 **Xử lí các icon**  
    - 👍 **Phân loại positive word**  
    - 👎 **Phân loại negative word**  
    - 😐 **Phân loại neutral word**  
    - ...  
    """)

    # Show another sample from df_summary
    st.write("### Dữ liệu sau khi đã tiền xử lí dữ liệu")
    st.dataframe(df_summary.head(5))

elif choice == "EDA":
    # Title
    st.title("Exploratory Data Analysis (EDA)")
    
    # Overview of the dataset
    st.write("### Tổng quan")
    st.write("""
    Bộ dữ liệu được cung cấp:
    - **7 cột và 21575 dòng**
    - **Cột `noi_dung_binh_luan` có 901 dòng bị null**
    """)
    # Subsection: Rating Distribution
    st.write("### Phân bố số lượng đánh giá theo số sao")
    
    # Calculate the counts for each rating level
    rating_counts = df_goc.groupby('so_sao')['so_sao'].count()
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
    ax.set_xlabel('Số sao')
    ax.set_ylabel('Số lượng đánh giá')
    ax.set_title('Phân bố số lượng đánh giá theo số sao')
    
    # Add value annotations on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    # Adjust x-axis ticks to show integer values only
    ax.set_xticks(rating_counts.index)

    # Show the chart in Streamlit
    st.pyplot(fig)

    # Display detailed statistics
    st.write("### Thống kê chi tiết số lượng đánh giá")
    stats_df = pd.DataFrame({
        'Số lượng đánh giá': rating_counts,
        'Tỷ lệ %': (rating_counts / rating_counts.sum() * 100).round(2)
    })
    st.dataframe(stats_df)

    # Additional statistics
    st.write(f"**Tổng số đánh giá:** {rating_counts.sum():,}")
    st.write(f"**Điểm trung bình:** {(rating_counts * rating_counts.index).sum() / rating_counts.sum():.2f}")
    st.write(f"**Mức đánh giá phổ biến nhất:** {rating_counts.idxmax()} sao ({rating_counts.max():,} đánh giá)")

    # Insights
    st.markdown("""
    ### Nhận xét:
    - Biểu đồ cho thấy sự hài lòng rất cao của khách hàng với sản phẩm/dịch vụ, thể hiện qua tỷ lệ đánh giá 5 sao vượt trội.
    - Tuy nhiên, tỷ lệ đánh giá từ 1-5 sao bị chênh lệch rất lớn => dữ liệu mất cân bằng.
    """)
    # Subsection: Customer Classification
    st.write("### Phân loại khách hàng theo số lượng đánh giá")

    # Define the categorization function
    def categorize_customer(n_reviews):
        if n_reviews == 1:
            return 'Một lần'
        elif n_reviews <= 3:
            return 'Ít (2-3)'
        elif n_reviews <= 5:
            return 'Trung bình (4-5)'
        else:
            return 'Nhiều (>5)'

    # Apply categorization
    customer_reviews = df_goc.groupby('ma_khach_hang').size().reset_index()
    customer_reviews.columns = ['ma_khach_hang', 'so_luong_danh_gia']
    customer_reviews['nhom_khach_hang'] = customer_reviews['so_luong_danh_gia'].apply(categorize_customer)

    # Plot the customer classification chart
    st.write("#### Biểu đồ phân loại khách hàng")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=customer_reviews,
        x='nhom_khach_hang',
        order=['Một lần', 'Ít (2-3)', 'Trung bình (4-5)', 'Nhiều (>5)'],
        palette="coolwarm",
        ax=ax
    )
    ax.set_title('Phân loại khách hàng theo số lượng đánh giá')
    ax.set_xlabel('Nhóm khách hàng')
    ax.set_ylabel('Số lượng khách hàng')
    ax.tick_params(axis='x', rotation=45)

    # Display the chart
    st.pyplot(fig)

    # Observations
    st.markdown("""
    ### Nhận xét:
    - Khách hàng đánh giá 2-3 lần và >5 lần cao => khách hàng trung thành.
    - Khách hàng chỉ đánh giá 1 lần: cần tập trung phân tích.
    """)
    # Subsection: Distribution of Customer Reviews
    st.write("### Phân bố số lượng đánh giá theo khách hàng")
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=customer_reviews, x='so_luong_danh_gia', bins=30, kde=True, color='blue', ax=ax)
    ax.set_title('Phân bố số lượng đánh giá theo khách hàng', fontsize=14)
    ax.set_xlabel('Số lượng đánh giá', fontsize=12)
    ax.set_ylabel('Số lượng khách hàng', fontsize=12)

    # Display the histogram
    st.pyplot(fig)

    # Basic Statistics
    total_customers = len(customer_reviews)
    avg_reviews = customer_reviews['so_luong_danh_gia'].mean()
    max_reviews = customer_reviews['so_luong_danh_gia'].max()
    min_reviews = customer_reviews['so_luong_danh_gia'].min()

    st.write("### Thống kê cơ bản về số lượng đánh giá của khách hàng")
    st.write(f"- **Tổng số khách hàng:** {total_customers:,}")
    st.write(f"- **Trung bình số đánh giá/khách:** {avg_reviews:.2f}")
    st.write(f"- **Số đánh giá cao nhất của một khách:** {max_reviews}")
    st.write(f"- **Số đánh giá thấp nhất của một khách:** {min_reviews}")

    # Insights
    st.markdown("""
    ### Nhận xét:
    - Hầu hết khách hàng chỉ đưa ra một số lượng đánh giá nhỏ.
    - Phần lớn các khách hàng chỉ thực hiện từ 1-10 đánh giá.
    """)

    # 1. Statistics by Year
    yearly_stats = df_summary.groupby('nam').agg({
        'ma_khach_hang': 'count',
        'so_sao': 'mean'
    }).round(2)
    yearly_stats.columns = ['Số lượng đánh giá', 'Điểm trung bình']
    yearly_stats = yearly_stats.sort_values('Số lượng đánh giá', ascending=False)

    # 2. Statistics by Month
    monthly_stats = df_summary.groupby('thang').agg({
        'ma_khach_hang': 'count',
        'so_sao': 'mean'
    }).round(2)
    monthly_stats.columns = ['Số lượng đánh giá', 'Điểm trung bình']
    monthly_stats = monthly_stats.sort_values('Số lượng đánh giá', ascending=False)

    # 3. Statistics by Weekday
    df_summary['thu'] = pd.to_datetime(df_summary['ngay_binh_luan']).dt.day_name()
    weekday_stats = df_summary.groupby('thu').agg({
        'ma_khach_hang': 'count',
        'so_sao': 'mean'
    }).round(2)
    weekday_stats.columns = ['Số lượng đánh giá', 'Điểm trung bình']
    weekday_stats = weekday_stats.sort_values('Số lượng đánh giá', ascending=False)

    # Display statistics
    st.write("#### Thống kê theo năm:")
    st.dataframe(yearly_stats)

    st.write("#### Thống kê theo tháng:")
    st.dataframe(monthly_stats)

    st.write("#### Thống kê theo ngày trong tuần:")
    st.dataframe(weekday_stats)

    # Visualization
    st.write("### Biểu đồ thống kê")

    # Visualization by Year
    st.write("#### Số lượng và điểm trung bình đánh giá theo năm")
    fig_year, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    yearly_stats['Số lượng đánh giá'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title("Số lượng đánh giá theo năm")
    ax1.set_xlabel("Năm")
    ax1.set_ylabel("Số lượng đánh giá")
    ax1.tick_params(axis='x', rotation=45)

    yearly_stats['Điểm trung bình'].plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title("Điểm trung bình theo năm")
    ax2.set_xlabel("Năm")
    ax2.set_ylabel("Điểm trung bình")
    ax2.tick_params(axis='x', rotation=45)

    st.pyplot(fig_year)

    # Visualization by Month
    st.write("#### Số lượng và điểm trung bình đánh giá theo tháng")
    fig_month, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 10))
    monthly_stats['Số lượng đánh giá'].plot(kind='bar', ax=ax3, color='skyblue')
    ax3.set_title("Số lượng đánh giá theo tháng")
    ax3.set_xlabel("Tháng")
    ax3.set_ylabel("Số lượng đánh giá")
    ax3.tick_params(axis='x', rotation=45)

    monthly_stats['Điểm trung bình'].plot(kind='bar', ax=ax4, color='orange')
    ax4.set_title("Điểm trung bình theo tháng")
    ax4.set_xlabel("Tháng")
    ax4.set_ylabel("Điểm trung bình")
    ax4.tick_params(axis='x', rotation=45)

    st.pyplot(fig_month)

    # Visualization by Weekday
    st.write("#### Số lượng và điểm trung bình đánh giá theo ngày trong tuần")
    fig_weekday, ax5 = plt.subplots(figsize=(10, 6))
    weekday_stats['Số lượng đánh giá'].plot(kind='bar', ax=ax5, color='skyblue')
    ax5.set_title("Số lượng đánh giá theo ngày trong tuần")
    ax5.set_xlabel("Thứ")
    ax5.set_ylabel("Số lượng đánh giá")
    ax5.tick_params(axis='x', rotation=45)

    st.pyplot(fig_weekday)

    # Key Insights
    st.write("### Nhận xét:")
    st.markdown(f"""
    - Năm có nhiều đánh giá nhất: **{yearly_stats.index[0]}** ({yearly_stats.iloc[0, 0]:,.0f} đánh giá).
    - Năm có điểm trung bình cao nhất: **{yearly_stats['Điểm trung bình'].idxmax()}** ({yearly_stats['Điểm trung bình'].max():.2f} sao).
    - Tháng có nhiều đánh giá nhất: **Tháng {monthly_stats.index[0]}** ({monthly_stats.iloc[0, 0]:,.0f} đánh giá).
    - Tháng có điểm trung bình cao nhất: **Tháng {monthly_stats['Điểm trung bình'].idxmax()}** ({monthly_stats['Điểm trung bình'].max():.2f} sao).
    - Ngày có nhiều đánh giá nhất: **{weekday_stats.index[0]}** ({weekday_stats.iloc[0, 0]:,.0f} đánh giá).
    - Ngày có điểm trung bình cao nhất: **{weekday_stats['Điểm trung bình'].idxmax()}** ({weekday_stats['Điểm trung bình'].max():.2f} sao).
    """)

elif choice == "Modeling - Evaluation":
    # Title
    st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">Modeling - Evaluation</h1>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <h2 style="color: #FF4500;">Tổng quan</h2>
    <p>Sau khi xử lý <b>TF-IDF</b> với feature <code>new_content</code>, thực hiện phân tích và dự đoán với các output:</p>
    <ul>
        <li><b>positive</b></li>
        <li><b>negative</b></li>
        <li><b>neutral</b></li>
    </ul>
    <p>Xử lý mất cân bằng dữ liệu bằng phương pháp <b>OverSampling</b> để cân bằng giữa các nhãn đầu ra.</p>
    """, unsafe_allow_html=True)

    # Section: Machine Learning Models
    st.markdown("""
    <h2 style="color: #4CAF50;">Các model Machine Learning được sử dụng:</h2>
    """, unsafe_allow_html=True)
    models = [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "AdaBoost",
        "XGBoost",
        "Support Vector Machine (Linear Kernel)",
        "Support Vector Machine (Polynomial Kernel)"
    ]
    st.write(", ".join(models))
    
    # Section: Performance Comparison Table
    st.markdown("""
    <h2 style="color: #FF4500;">Bảng so sánh hiệu suất giữa các mô hình</h2>
    """, unsafe_allow_html=True)
    compare_path = "image/model_compare.png"  # Update to the correct path if necessary
    st.image(compare_path, caption="Bảng so sánh hiệu suất các mô hình", use_container_width=True)

    # Section: Insights and Model Selection
    st.markdown("""
    <h2 style="color: #4CAF50;">Nhận xét và chọn mô hình</h2>
    """, unsafe_allow_html=True)
    st.markdown("""
    - Dựa vào bảng so sánh, mô hình <b>ExtraTreesClassifier</b> được chọn vì:
        - Độ chính xác trên tập test (<code>accuracy_test</code>) và tập train (<code>accuracy_train</code>) đều cao nhất.
        - Sự chênh lệch giữa độ chính xác trên tập train và test không quá lớn (<b>≈0.45%</b>).
        - Thời gian chạy nhanh (<b>36s</b>), phù hợp cho ứng dụng thực tế.
    """, unsafe_allow_html=True)

    # Section: Confusion Matrix
    st.markdown("""
    <h2 style="color: #FF4500;">Confusion Matrix</h2>
    """, unsafe_allow_html=True)
    confusion_path = "image/confusion_matrix.png"  # Update to the correct path if necessary
    st.image(confusion_path, use_container_width=True)

    # ROC and Precision-Recall Curves
    st.markdown("""
    <h2 style="color: #4CAF50;">ROC-AUC and Precision-Recall Curves</h2>
    """, unsafe_allow_html=True)
    roc_path = "image/roc.png"  # Update to the correct path
    st.image(roc_path, caption="ROC-AUC and Precision-Recall Curves", use_container_width=True)

    # Insights about the curves
    st.markdown("""
    <h2 style="color: #FF4500;">Nhận xét:</h2>
    <ul>
        <li><b>ROC-AUC Curve:</b>
            <ul>
                <li>Mô hình đạt được AUC cao (gần <b>0.97-0.98</b>) ở cả ba lớp, chứng tỏ mô hình phân biệt rất tốt giữa các nhãn.</li>
                <li>Đường cong nằm gần góc trên bên trái, cho thấy mô hình hoạt động hiệu quả về việc giảm False Positives và tăng True Positives.</li>
            </ul>
        </li>
        <li><b>Precision-Recall Curve:</b>
            <ul>
                <li>Mô hình đạt độ chính xác cao (precision) trên lớp 2 (AUC = <b>1.0</b>), nhưng thấp hơn với lớp 1 và lớp 0.</li>
                <li>Với lớp có AUC = <b>0.8</b> (Class 1), độ chính xác giảm dần khi tỷ lệ hồi đáp (recall) tăng.</li>
                <li>Điều này cho thấy cần kiểm tra thêm các lớp dữ liệu để giảm thiểu việc đánh đổi giữa precision và recall.</li>
            </ul>
        </li>
    </ul>
    <h3 style="color: #FF4500;">Tổng quan:</h3>
    <p>- Mô hình <b>ExtraTreesClassifier</b> có khả năng phân loại mạnh mẽ, đặc biệt trên các nhãn chính xác cao (Class 2).</p>
    <p>- Precision-Recall Curves gợi ý rằng có thể cải thiện precision cho các lớp thấp hơn.</p>
    """, unsafe_allow_html=True)




elif choice == "Comment Sentiment Analysis":
    # Title and logo
    logo_hasaki_path = "image/hasaki_banner.jpg"  # Replace with the path to your banner/logo
    st.image(logo_hasaki_path, width=700)
    st.title("Comment Sentiment Analysis")

    # Choose between entering a single comment manually or loading from a file
    analysis_option = st.radio(
        "Chọn phương thức nhập liệu:",
        ("Nhập một comment", "Tải file danh sách comment (.txt)")
    )

    if analysis_option == "Nhập một comment":
        st.subheader("Nhập comment cần phân tích:")
        input_text = st.text_area("Nhập bình luận cần phân tích:")
        
        if st.button("Phân tích", key="single"):
            if input_text:
                with st.spinner("Đang phân tích..."):
                    result = predict_sentiment(input_text, vectorizer, model, loaded_le)
                    if 'error' not in result:
                        st.success(f"Cảm xúc: {result['sentiment']}")
                    else:
                        st.error(f"Lỗi: {result['error']}")
            else:
                st.warning("Vui lòng nhập bình luận cần phân tích")

    elif analysis_option == "Tải file danh sách comment (.txt)":
        st.subheader("Tải file chứa danh sách comment (định dạng .txt):")

        # Thêm upload file
        uploaded_file = st.file_uploader(
            "Kéo thả hoặc chọn file text chứa bình luận (mỗi dòng một bình luận)",
            type=['txt']
        )
        
        # Text area cho nhập trực tiếp
        input_texts = st.text_area(
            "Nhập danh sách bình luận (mỗi bình luận một dòng):",
            height=200
        )

        if st.button("Phân tích", key="batch"):
            texts = []
            
            # Xử lý file upload nếu có
            if uploaded_file is not None:
                text_content = uploaded_file.getvalue().decode('utf-8')
                texts.extend([line.strip() for line in text_content.split('\n') if line.strip()])
                
            # Thêm text từ text area nếu có 
            if input_texts:
                texts.extend([text.strip() for text in input_texts.split('\n') if text.strip()])

            if texts:
                results = []
                
                progress_bar = st.progress(0)
                for i, text in enumerate(texts):
                    result = predict_sentiment(text, vectorizer, model, loaded_le)
                    if 'error' not in result:
                        results.append({
                            'text': text,
                            'sentiment': result['sentiment'],
                            **result['features']
                        })
                    progress_bar.progress((i + 1) / len(texts))
                
                if results:
                    df = pd.DataFrame(results)
                    st.success(f"Đã phân tích {len(results)} bình luận")
                    
                    # Thống kê
                    st.subheader("Thống kê")
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment_counts = df['sentiment'].value_counts()
                        st.bar_chart(sentiment_counts)
                    
                    with col2:
                        st.write("Phân bố cảm xúc:")
                        for sentiment, count in sentiment_counts.items():
                            st.write(f"{sentiment}: {count} ({count/len(df)*100:.1f}%)")
                    
                    # Kết quả chi tiết
                    st.subheader("Kết quả chi tiết")
                    st.dataframe(df)
                    
                    # Tải xuống kết quả
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Tải xuống kết quả (CSV)",
                        csv,
                        "sentiment_analysis_results.csv",
                        "text/csv"
                    )
                else:
                    st.error("Không thể phân tích bình luận nào")
            else:
                st.warning("Vui lòng nhập hoặc tải lên file chứa bình luận cần phân tích")


elif choice == "Product Sentiment Analysis":
    # Title
    st.title("Product Sentiment Analysis")
    st.subheader("Chọn sản phẩm để xem đánh giá")

    # Check if data is loaded
    if 'df_summary' in locals() and not df_summary.empty:
        # Randomly select 30 products
        random_products = df_summary.sample(n=30, random_state=42)[['ten_san_pham', 'ma_san_pham']]

        # Create product options as tuples of (name, ID)
        product_options = [(row['ten_san_pham'], row['ma_san_pham']) for _, row in random_products.iterrows()]

        # Selectbox for product selection
        selected_product = st.selectbox(
            "Chọn sản phẩm",
            options=product_options,
            format_func=lambda x: x[0]  # Display only the product name
        )

        # Display selected product details
        st.write("### Bạn đã chọn:")
        st.write(f"**Tên sản phẩm:** {selected_product[0]}")
        st.write(f"**Mã sản phẩm:** {selected_product[1]}")

        # Analysis and visualization options
        if selected_product:
            product_id = selected_product[1]  # Extract product ID
            product_data = df_summary[df_summary['ma_san_pham'] == product_id]

        if not product_data.empty:
            # Display basic product information
            product_name = product_data['ten_san_pham'].iloc[0]
            avg_rating = product_data['so_sao'].mean()
            total_reviews = len(product_data)
            unique_customers = product_data['ma_khach_hang'].nunique()

            # Count sentiment categories
            sentiment_counts = product_data['output'].value_counts()
            positive_reviews = sentiment_counts.get('positive', 0)
            neutral_reviews = sentiment_counts.get('neutral', 0)
            negative_reviews = sentiment_counts.get('negative', 0)
            # Display product information
            st.write(f"**Điểm đánh giá trung bình:** {avg_rating:.2f}")
            st.write(f"**Tổng số nhận xét:** {total_reviews}")
            st.write(f"**Số lượng mã khách hàng duy nhất:** {unique_customers}")
            st.write(f"**Số nhận xét tích cực:** {positive_reviews}")
            st.write(f"**Số nhận xét trung tính:** {neutral_reviews}")
            st.write(f"**Số nhận xét tiêu cực:** {negative_reviews}")
            display_wordclouds(product_data)
            # Choose the filter option
            st.write("### Tùy chọn phân tích:")
            filter_option = st.selectbox("Chọn cách hiển thị:", ["Tất cả các năm", "Theo tháng"])

            # Additional dropdown for year/month selection
            selected_year = None
            selected_month = None
            if filter_option == "Theo tháng":
                selected_year = st.selectbox("Chọn năm:", sorted(product_data['nam'].unique()))

            # Call the function to display charts
            display_analysis_charts(product_data, product_name, filter_option, selected_year)
        else:
            st.warning("Không tìm thấy dữ liệu cho sản phẩm này.")
    else:
        st.warning("Không thể tải dữ liệu từ file df_summary.csv. Vui lòng kiểm tra file dữ liệu.")



def display_team_members_in_sidebar(members):
    st.sidebar.markdown('<h2 style="color:#0047AB; text-align: left;">Thành viên nhóm:</h2>', unsafe_allow_html=True)
    member_html = '<div style="color:#0047AB; font-size: 18px; text-align: left; font-weight: bold;">' + \
                  '<br>'.join(members) + \
                  '</div>'
    st.sidebar.markdown(member_html, unsafe_allow_html=True)

team_members = ["1. Phan Minh Huệ", "2. Huỳnh Danh Nhân"]
display_team_members_in_sidebar(team_members)