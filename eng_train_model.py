import pandas as pd
from sklearn.model_selection import train_test_split


# kaggle에 있는 labeled IMDB data
data = pd.read_csv('labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)

train_data, test_data = train_test_split(data,test_size=0.2,random_state=1)

train_data.drop_duplicates(subset=['review'], inplace=True) # 중복값 제거
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인


test_data.drop_duplicates(subset=['review'], inplace=True) # 중복값 제거
test_data = test_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(test_data.isnull().values.any()) # Null 값이 존재하는지 확인

print(len(train_data))
print(len(test_data))

print(train_data.groupby('sentiment').size().reset_index(name = 'count'))

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

# review 전처리 함수
def review_to_words(raw):
    new= re.sub('[^a-zA-Z]',' ',raw)  # 알파벳을 제외한 나머지 글자 제거
    words = new.lower().split()   # 소문자로 변경
    stop_words = set(stopwords.words('english'))  # 불용어 집합
    meaning_words = [w for w in words if not w in stop_words] # 불용어 제외
    words = [wordnet.lemmatize(w) for w in meaning_words]   # 표제어 추출
    words = [w for w in words if len(w) > 2]  # 단어길이가 3 이상인 단어만 저장
    return words

# train_data['review']를 전처리 하기 위한 반복문
X_train = []

for i in range(len(train_data['review'])):
    n = train_data.index[i]
    X_train.append(review_to_words(train_data['review'][n]))

print(X_train[:5])


# test_data['review']를 전처리 하기 위한 반복문
X_test = []

for i in range(len(test_data['review'])):
    n = test_data.index[i]
    X_test.append(review_to_words(test_data['review'][n]))

print(X_test[:5])

from tensorflow.keras.preprocessing.text import Tokenizer

# vocab_size를 위한 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

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

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)

# 정수인코딩
tokenizer = Tokenizer(vocab_size,oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# tokenizer 저장
import pickle
with open('tokenizer_data_eng.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import numpy as np
y_train = np.array(train_data['sentiment'])
y_test = np.array(test_data['sentiment'])

# 빈 샘플 인덱스
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# 빈 샘플 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))


# 샘플 댓글길이의 비율
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 300

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 패딩
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import os

model = Sequential()
model.add(Embedding(vocab_size, 256))
model.add(Dropout(0.3))
model.add(Conv1D(256, 3, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)

checkpoint_path = "eng_ckpt/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

mc = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,
                     verbose=1,monitor='val_acc', mode='max',save_best_only=True)

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 15, validation_data = (X_test, y_test), callbacks=[es, mc])

model.save('eng_model.h5')

model = load_model('eng_model.h5')

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

