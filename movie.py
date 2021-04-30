from selenium import webdriver
from bs4 import BeautifulSoup


movie_title = input('영화 제목을 입력하세요>>>')


path = (r"C:\Users\STU\Downloads\chromedriver_win32\chromedriver")
driver = webdriver.Chrome(path)

driver.get('https://www.naver.com')

search_box = driver.find_element_by_name('query')
search_box.send_keys(f'영화 {movie_title}')
search_box.submit()

a = driver.find_elements_by_xpath('//*[@id="main_pack"]/div[1]/div[1]/div[1]/h2/a')
driver.get(a[0].get_attribute('href'))

req = driver.page_source

soup=BeautifulSoup(req, 'html.parser')

selector = '#content > div.article > div.mv_info_area > div.mv_info > h3 > a'
selector_2 = '#content > div.article > div.mv_info_area > div.mv_info > strong'

default = soup.select(selector_2)
for a in default:
    eng_title = a.text
print(eng_title)

links = soup.select(selector)
c = []
for link in links:
    c.append(link['href'])

code = c[0][-6:]

if '=' in code:
    code = c[0][-5:]

print(code)

driver.get('https://www.google.com')

search_box = driver.find_element_by_name('q')
search_box.send_keys(f'{eng_title}')
search_box.submit()

a = driver.find_elements_by_xpath('//*[@id="kp-wp-tab-overview"]/div[1]/div[2]/div/div/div[1]/div[1]/a[2]')
driver.get(a[0].get_attribute('href'))

a = driver.find_elements_by_xpath('//*[@id="criticHeaders"]/a[1]')
driver.get(a[0].get_attribute('href'))

url = driver.current_url

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('eng_model.h5')

eng_pos_num = 0
eng_neg_num = 0
eng_sum_score = 0
eng_pos_dict = dict()
eng_neg_dict = dict()

p_eng=[]
n_eng=[]

max_len = 300

import re
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

import pickle

with open('tokenizer_data_eng.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# def sentiment_predict_eng(new_sentence):
#     new = re.sub('[^a-zA-Z]', ' ', new_sentence)
#     words = new.lower().split()
#     stop_words = set(stopwords.words('english'))
#     meaning_words = [w for w in words if not w in stop_words]
#     words = [wordnet.lemmatize(w) for w in meaning_words]
#     words = [w for w in words if len(w) > 2]
#     new = tokenizer.texts_to_sequences([words])
#     eng_texts_test = pad_sequences(new, maxlen=max_len)
#     score = float(model.predict(eng_texts_test)) # 예측
#     if(score > 0.5):
#       p_eng.append(words)
#     else:
#       n_eng.append(words)
#
# import requests
#
# def get_data_eng(url):
#     response = requests.get(url)
#     html = response.text.strip()
#
#     soup = BeautifulSoup(html, 'html.parser')
#     selector = '#content > div > div > div > div.review_table > div > div.col-xs-16.review_container > div.review_area > div.review_desc > div.the_review'
#     links = soup.select(selector)
#     for link in links:
#         t = link.text.strip()
#         sentiment_predict_eng(t)
#
#
# get_data_eng(html)
#
# print(p_eng)
# print(n_eng)
#
#
# p_eng = [element for array in p_eng for element in array]
# n_eng = [element for array in n_eng for element in array]


import requests

for i in range(1, 6):
    new_url = url + f'?type=&sort=&page={i}'
    response = requests.get(new_url)
    html = response.text.strip()

    soup = BeautifulSoup(html, 'html.parser')
    selector = '#content > div > div > div > div.review_table > div > div.col-xs-16.review_container > div.review_area > div.review_desc > div.the_review'
    links = soup.select(selector)

    for link in links:
        eng_text = link.text.strip()
        new = re.sub('[^a-zA-Z]', ' ', eng_text)
        words = new.lower().split()
        stop_words = set(stopwords.words('english'))
        meaning_words = [w for w in words if not w in stop_words]
        words = [wordnet.lemmatize(w) for w in meaning_words]
        words = [w for w in words if len(w) > 2]
        new = tokenizer.texts_to_sequences([words])
        eng_texts_test = pad_sequences(new, maxlen=max_len)
        eng_score = float(model.predict(eng_texts_test))  # 예측
        eng_sum_score += eng_score
        if (eng_score > 0.5):
            eng_pos_num += 1
            p_eng.append(eng_text)

        else:
            eng_neg_num += 1
            n_eng.append(eng_text)

import time
from konlpy.tag import Okt

okt = Okt()
kr_stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

driver.implicitly_wait(10)
driver.get('https://papago.naver.com/?sk=en&tk=ko&hn=0')

new_list_p =[]
new_list_n =[]

search = driver.find_element_by_xpath("/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").send_keys(p_eng)
time.sleep(2)
# 번역 버튼
button = driver.find_element_by_css_selector("#btnTranslate > span.translate_pc___2dgT8").click()

# 번역창에 뜬 결과물 가져오기
result = driver.find_element_by_css_selector("#txtTarget > span").text

# 결과물 전처리
tokenized_sentence = okt.pos(result, stem=True)  # 토큰화
exst_tok_sentence = [word[0] for word in tokenized_sentence if not word[0] in kr_stopwords] # 불용어 제거

new_list_p.append(exst_tok_sentence)

driver.implicitly_wait(10)
driver.get('https://papago.naver.com/?sk=en&tk=ko&hn=0')

# 검색창에 영어 문장/단어 넣기 (부정적)
search_n = driver.find_element_by_xpath("/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").send_keys(n_eng)
time.sleep(2)
# 번역 버튼
button_n = driver.find_element_by_css_selector("#btnTranslate > span.translate_pc___2dgT8").click()

# 번역창에 뜬 결과물 가져오기
result_n = driver.find_element_by_css_selector("#txtTarget > span").text

# 결과물 전처리
tokenized_sentence_n = okt.pos(result_n, stem=True)  # 토큰화
exst_tok_sentence_n = [word[0] for word in tokenized_sentence_n if not word[0] in kr_stopwords] # 불용어 제거


new_list_n.append(exst_tok_sentence_n)

# dict 만들기
p_counts = dict()
n_counts = dict()

for i in range(len(new_list_p)):
    for word in new_list_p[i]:
        if word in p_counts:
            p_counts[word] += 1
        else:
            p_counts[word] = 1

for i in range(len(new_list_n)):
    for word in new_list_n[i]:
        if word in n_counts:
            n_counts[word] += 1
        else:
            n_counts[word] = 1

p_counts = dict(sorted(p_counts.items(),reverse=True,key=lambda item: item[1]))
n_counts = dict(sorted(n_counts.items(),reverse=True,key=lambda item: item[1]))

print(p_counts)
print(n_counts)
