
from konlpy.tag import Okt
from stylecloud import stylecloud

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

with open('tokenizer_data_ko.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

vocab_size = total_cnt - rare_cnt + 2

from tensorflow.keras.models import load_model

model = load_model('ko_model.h5')

checkpoint_path = "ko_ckpt/cp.ckpt"
model.load_weights(checkpoint_path)

okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
max_len = 30

p = []
n = []

def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(model.predict(pad_new)) # 예측
  if(score > 0.5):
      p.append(new_sentence)
  else:
      n.append(new_sentence)

import requests

def get_data(url):
    resp = requests.get(url)
    html = BeautifulSoup(resp.content, 'html.parser')
    score_result = html.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')
    for i, li in enumerate(lis):
        review_text = li.find('p').getText()  # span id = _filtered_ment_0
        review_text = review_text.replace("관람객","")
        review_text = review_text.strip()
        score = li.find('em').getText()

        sentiment_predict(review_text)


from selenium import webdriver
from bs4 import BeautifulSoup


movie_title = input('영화 제목을 입력하세요>>>')


path = ("chromedriver")
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

links = soup.select(selector)
c = []
for link in links:
    c.append(link['href'])

code = c[0][-6:]

if '=' in code:
    code = c[0][-5:]

#
# test_url = f'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code={code}&type=after'
# resp = requests.get(test_url)
# html = BeautifulSoup(resp.content, 'html.parser')
# # result = html.find('div', {'class':'score_total'}).find('strong').findChildren('em')[0].getText()
# total_count = 100 #int(result.replace(',', ''))

#
# for i in range(1, int(total_count / 10) + 1):
#     url = test_url + '&page=' + str(i)
#     print('url: "' + url + '" is parsing....')
#     get_data(url)
#
#
# p_counts = dict()
# n_counts = dict()
#
# for i in range(len(p)):
#     for word in p[i]:
#         if word in p_counts:
#             p_counts[word] += 1
#         else:
#             p_counts[word] = 1
#
#
# for i in range(len(n)):
#     for word in n[i]:
#         if word in n_counts:
#             n_counts[word] += 1
#         else:
#             n_counts[word] = 1
#
#
# p_counts = sorted(p_counts.items(),reverse=True,key=lambda item: item[1])
#
# n_counts = sorted(n_counts.items(),reverse=True,key=lambda item: item[1])
#
# print(p[:100])
# print(n[:100])

total_count = 100  # int(result.replace(',', ''))
sum_score = 0
pos_num = 0
neg_num = 0
pos_dict = dict()
neg_dict = dict()

for i in range(1, int(total_count / 10) + 1):
    url = (f'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code={code}&type=after&page=' + str(i))
    print(f'{url} is parsing...')
    resp = requests.get(url)
    html = BeautifulSoup(resp.content, 'html.parser')
    score_result = html.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')

    for li in lis:
        review_text = li.find('p').getText()  # span id = _filtered_ment_0
        review_text = review_text.replace("관람객", "")
        review_text = review_text.strip()
        score = int(li.find('em').getText())
        sum_score += score

        tokenized_sentence = okt.pos(review_text, stem=True)  # 토큰화
        exstopw_ts = [word for word in tokenized_sentence if not word[0] in stopwords]  # 불용어 제거
        exst_tok_sentence = [word[0] for word in tokenized_sentence if not word[0] in stopwords]
        encoded = tokenizer.texts_to_sequences([exst_tok_sentence])  # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
        pd_score = float(model.predict(pad_new))  # 예측
        if (pd_score > 0.5):
            pos_num += 1
            for word in exstopw_ts:
                if word[1] in ['Noun', 'Adjective', 'Verb']:
                    if word[0] not in pos_dict:
                        pos_dict[word[0]] = 1
                    else:
                        pos_dict[word[0]] += 1
        else:
            neg_num += 1
            for word in exstopw_ts:
                if word[1] in ['Noun', 'Adjective', 'Verb']:
                    if word[0] not in neg_dict:
                        neg_dict[word[0]] = 1
                    else:
                        neg_dict[word[0]] += 1

avg_score = sum_score / total_count
pos_dict = dict(sorted(pos_dict.items(), reverse=True, key=lambda item: item[1]))
neg_dict = dict(sorted(neg_dict.items(), reverse=True, key=lambda item: item[1]))

print(avg_score)
print(pos_dict)
print(neg_dict)



# PNG file create --------------------------------------------
wcstopwords = {'영화', '보다', '되다', '있다', '없다', '아니다', '이다', '좋다', '않다', '같다',
                       '많다', '때', '것', '바', '그', '수'}
for w in wcstopwords:
    if w in pos_dict:
        pos_dict.pop(w)
    if w in neg_dict:
        neg_dict.pop(w)

# stylecloud part
stylecloud.gen_stylecloud(text=pos_dict,    # 긍정 리뷰 사전
                                  font_path='C:/Windows/Fonts/BMJUA_ttf.ttf',   # 폰트
                                  icon_name="fas fa-carrot",    # 당근
                                  palette="cartocolors.sequential.Peach_5",     # 주황~분홍
                                  background_color='black',     # 배경
                                  output_name="positive_2.png"
                                  )
stylecloud.gen_stylecloud(text=neg_dict,
                                  font_path='C:/Windows/Fonts/BMJUA_ttf.ttf',
                                  icon_name="fas fa-bomb",
                                  palette="colorbrewer.sequential.YlGn_4",
                                  background_color='black',
                                  output_name="negative_2.png"
                                  )

