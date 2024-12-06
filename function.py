import os
import nltk
# nltk.download('punkt')
os.getcwd()



# LOAD EMOJICON
# file = open('files/emojicon.txt', 'r', encoding="utf8")
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
# LOAD TEENCODE
# file = open('files/teencode.txt', 'r', encoding="utf8")
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
# LOAD TRANSLATE ENGLISH -> VNMESE
# file = open('files/english-vnmese.txt', 'r', encoding="utf8")
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
##################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
# LOAD STOPWORDS
# file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()
# LOAD POSITIVE WORDS
# file = open('files/positive_words.txt', 'r', encoding="utf8")
file = open('files/positive_words.txt', 'r', encoding="utf8")
positive_lst = file.read().split('\n')
file.close()
# LOAD NEGATIVE WORDS
# file = open('files/negative_words.txt', 'r', encoding="utf8")
file = open('files/negative_words.txt', 'r', encoding="utf8")
negative_lst = file.read().split('\n')
file.close()
# LOAD NEUTRAL_WORDS
# file = open('files/negative_words.txt', 'r', encoding="utf8")
file = open('files/neutral_words.txt', 'r', encoding="utf8")
neutral_words = file.read().split('\n')
file.close()
# LOAD NEGATIVE EMOJIS
# file = open('files/negative_words.txt', 'r', encoding="utf8")
file = open('files/negative_emojis.txt', 'r', encoding="utf8")
negative_emojis = file.read().split('\n')
file.close()# LOAD POSITIVE EMOJIS
# file = open('files/negative_words.txt', 'r', encoding="utf8")
file = open('files/positive_emojis.txt', 'r', encoding="utf8")
positive_emojis = file.read().split('\n')
file.close()

import re
import regex
from nltk.tokenize import sent_tokenize, word_tokenize
from pyvi.ViTokenizer import tokenize
# Danh sách các từ mang ý nghĩa phủ định
negation_words = ["không", "nhưng", "tuy nhiên", "mặc dù", "chẳng", "mà", 'kém', 'giảm']
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
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from pyvi.ViTokenizer import tokenize
from pyvi.ViTokenizer import*
# Định nghĩa hàm tiền xử lý
def preprocess_input(input_text, emoji_dict, teen_dict, wrong_lst, neutral_words, negative_words, positive_words, negation_words, positive_emojis, negative_emojis, stopwords_lst):
    """
    Tiền xử lý văn bản đầu vào từ người dùng, trích xuất các đặc trưng cảm xúc.
    """
    # Bước 1: Xử lý văn bản
    processed_text = process_text(input_text, emoji_dict, teen_dict, wrong_lst)

    # Bước 2: Chuyển đổi Unicode
    processed_text = covert_unicode(processed_text)

    # Bước 3: Tính toán số lượng từ/emoji
    neutral_word_count = find_words_test(processed_text, neutral_words)[0]
    negative_word_count = find_words_test(processed_text, negative_words)[0]
    positive_word_count = max(find_words_test(processed_text, positive_words)[0] - find_words_test(processed_text, negation_words)[0], 0)
    positive_emoji_count = find_words_test(processed_text, positive_emojis)[0]
    negative_emoji_count = find_words_test(processed_text, negative_emojis)[0]

    # Bước 4: Loại bỏ stopwords
    tokenized_text = ViTokenizer.tokenize(processed_text)

    # Đảm bảo tokenized_text là chuỗi trước khi loại bỏ stopwords
    tokenized_text = remove_stopword(tokenized_text, stopwords_lst)

    processed_data = {
        "processed_text": tokenized_text,
        "neutral_word_count": neutral_word_count,
        "negative_word_count": negative_word_count,
        "positive_word_count": positive_word_count,
        "positive_emoji_count": positive_emoji_count,
        "negative_emoji_count": negative_emoji_count
    }
    return processed_data
def covert_unicode(txt):
    dicchar = loaddicchar()
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
def find_words_test(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in document_lower.split():  # Tách văn bản thành từng chữ
        if word in list_of_words:
            word_count += 1  # Tăng biến đếm mỗi khi tìm thấy một chữ trong list_of_words
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
########################################################################
with open('label_encoder.pkl', 'rb') as f:
    label = pickle.load(f)

import joblib
import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import LabelEncoder
def sentiment_prediction_pipeline(user_input):
    """
    Hàm tổng thể để dự đoán cảm xúc từ văn bản đầu vào

    Parameters:
    user_input (str): Văn bản cần phân tích

    Returns:
    str: Nhãn cảm xúc được dự đoán
    """
    try:
        # Load các models và resources cần thiết
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('ExtraTreesClassifier_model_v1.pkl')
        with open('label_encoder.pkl', 'rb') as f:
            loaded_le = pickle.load(f)

        # Tiền xử lý văn bản
        processed_data = preprocess_input(
            user_input,
            emoji_dict,
            teen_dict,
            wrong_lst,
            neutral_words,
            negative_lst,
            positive_lst,
            negation_words,
            positive_emojis,
            negative_emojis,
            stopwords_lst
        )

        # Véc tơ hóa văn bản
        processed_text = processed_data['processed_text']
        text_vectorized = vectorizer.transform([processed_text])

        # Tạo feature vector
        feature_values = [
            processed_data['neutral_word_count'],
            processed_data['negative_word_count'],
            processed_data['positive_word_count'],
            processed_data['positive_emoji_count'],
            processed_data['negative_emoji_count']
        ]

        # Kết hợp đặc trưng
        features_combined = hstack((
            text_vectorized,
            np.array(feature_values).reshape(1, -1)
        ))

        # Dự đoán
        prediction = model.predict(features_combined)
        predicted_label = loaded_le.inverse_transform(prediction)[0]

        return {
            'sentiment': predicted_label,
            'processed_text': processed_text,
            'features': {
                'neutral_words': processed_data['neutral_word_count'],
                'negative_words': processed_data['negative_word_count'],
                'positive_words': processed_data['positive_word_count'],
                'positive_emojis': processed_data['positive_emoji_count'],
                'negative_emojis': processed_data['negative_emoji_count']
            }
        }

    except Exception as e:
        return {
            'error': str(e),
            'sentiment': None,
            'processed_text': None,
            'features': None
        }

################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Function to display WordClouds
def display_wordclouds(product_data):
    custom_stopwords = set(['sản_phẩm', 'không'])
    positive_text = " ".join(product_data[product_data['output'] == 'positive']['new_content'])
    negative_text = " ".join(product_data[product_data['output'] == 'negative']['new_content'])
    neutral_text = " ".join(product_data[product_data['output'] == 'neutral']['new_content'])

    wordcloud_positive = WordCloud(width=400, height=400, background_color='white',stopwords=custom_stopwords).generate(positive_text) if positive_text else None
    wordcloud_negative = WordCloud(width=400, height=400, background_color='white',stopwords=custom_stopwords).generate(negative_text) if negative_text else None
    wordcloud_neutral = WordCloud(width=400, height=400, background_color='white',stopwords=custom_stopwords).generate(neutral_text) if neutral_text else None

    st.write("### WordClouds for Sentiments")
    cols = st.columns(3)

    with cols[0]:
        st.write("**Tích cực**")
        if wordcloud_positive:
            st.image(wordcloud_positive.to_array())
        else:
            st.write("Không có dữ liệu tích cực")

    with cols[1]:
        st.write("**Trung tính**")
        if wordcloud_neutral:
            st.image(wordcloud_neutral.to_array())
        else:
            st.write("Không có dữ liệu trung tính")

    with cols[2]:
        st.write("**Tiêu cực**")
        if wordcloud_negative:
            st.image(wordcloud_negative.to_array())
        else:
            st.write("Không có dữ liệu tiêu cực")

import matplotlib.pyplot as plt
import streamlit as st


def display_analysis_charts(product_data, product_name, filter_option, selected_year=None):
    """
    Trực quan hóa dữ liệu sản phẩm: phân bố sentiment, sentiment theo năm/tháng, và số lượng khách hàng.

    Args:
        product_data (pd.DataFrame): Dữ liệu sản phẩm.
        product_name (str): Tên sản phẩm.
        filter_option (str): Bộ lọc ("Tất cả các năm", "Theo tháng").
        selected_year (int, optional): Năm được chọn (nếu có).
    """
    # Apply filters based on the selected option
    if filter_option == "Tất cả các năm":
        filtered_data = product_data
        grouping_columns = ['nam', 'output']
        customer_grouping = 'nam'
        xlabel = "Năm"
        time_index = None
    elif filter_option == "Theo tháng":
        if selected_year is None:
            st.warning("Vui lòng chọn một năm để xem dữ liệu theo tháng.")
            return
        filtered_data = product_data[product_data['nam'] == selected_year]
        grouping_columns = ['thang', 'output']
        customer_grouping = 'thang'
        xlabel = "Tháng"
        time_index = range(1, 13)  # Tháng từ 1 đến 12

    # Aggregate data for charts
    sentiment_data = filtered_data.groupby(grouping_columns).size().unstack(fill_value=0)
    
    if time_index:
        # Bảo đảm rằng tất cả các tháng từ 1 đến 12 đều xuất hiện, ngay cả khi không có dữ liệu
        sentiment_data = sentiment_data.reindex(index=time_index, fill_value=0)

    time_periods = sentiment_data.index
    positive_counts = sentiment_data.get('positive', 0)
    neutral_counts = sentiment_data.get('neutral', 0)
    negative_counts = sentiment_data.get('negative', 0)

    # Number of unique customers
    customer_counts = filtered_data.groupby(customer_grouping)['ma_khach_hang'].nunique()

    if time_index:
        customer_counts = customer_counts.reindex(index=time_index, fill_value=0)

    # Chart 1: Overall Sentiment Distribution
    st.write("### Biểu đồ 1: Phân bố nhận xét tổng thể")
    sentiment_counts = filtered_data['output'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax1)
    ax1.set_xlabel("Loại nhận xét")
    ax1.set_ylabel("Số lượng")
    ax1.tick_params(axis='x', rotation=0)

    # Display the chart
    st.pyplot(fig1)

    # Chart 2: Sentiment Distribution by Time Period
    st.write(f"### Biểu đồ 2: Phân bố nhận xét theo {xlabel}")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(time_periods, positive_counts, color='green', label='Tích cực')
    ax2.bar(time_periods, neutral_counts, bottom=positive_counts, color='orange', label='Trung tính')
    ax2.bar(time_periods, negative_counts, bottom=positive_counts + neutral_counts, color='red', label='Tiêu cực')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Số lượng nhận xét")
    ax2.legend()

    # Display the chart
    st.pyplot(fig2)

    # Chart 3: Number of Customers by Time Period
    st.write(f"### Biểu đồ 3: Số khách hàng theo {xlabel}")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    customer_counts.plot(kind='bar', color='blue', ax=ax3)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Số lượng khách hàng")
    ax3.tick_params(axis='x', rotation=0)

    # Display the chart
    st.pyplot(fig3)




