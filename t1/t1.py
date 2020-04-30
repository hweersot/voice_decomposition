import numpy as np
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

def preprocessing(review,remove_stopwords):
    review=train_data['review'][0]
    review_text=BeautifulSoup(review,'html5lib').get_text()
    review_text=re.sub('[^a-zA-Z]',' ',review_text)
    if remove_stopwords==True:
        stop_words=set(stopwords.words('english'))
        review_text=review_text.lower()
        words=review_text.split()
        words=[w for w in words if w not in stop_words]
        clean_review=' '.join(words)
        return clean_review
    else:
        return review_text

d_path='../data_in/'
train_input_data='train_input.npy'
train_label_data='train.label.npy'
train_clean_data='train_clean.csv'
train_data=pd.read_csv(d_path+'/labeledTrainData.tsv',header=0,delimiter='\t',quoting=3)
clean_train_reviews=[]
cnt=0
for review in train_data['review']:
    cnt+=1
    clean_train_reviews.append(preprocessing(review,remove_stopwords=True))
print(clean_train_reviews[0])
clean_train_df=pd.DataFrame({'review':clean_train_reviews,'sentiment':train_data['sentiment']})
tokenizer=Tokenizer()
tokenizer.fit_on_texts(clean_train_reviews)
text_sequences=tokenizer.texts_to_sequences(clean_train_reviews)
word_vocab=tokenizer.word_index
train_inputs=pad_sequences(text_sequences,maxlen=174,padding='post')
train_labels=np.array(train_data['sentiment'])

#저장부
np.save(open(d_path+train_input_data,'wb'),train_inputs)
np.save(open(d_path+train_label_data,'wb'),train_labels)
clean_train_df.to_csv(d_path+train_clean_data,index=False)



d_path='../data_in/'
test_input_data='test_input.npy'
test_id_data='test_id.npy'
test_clean_data='test_clean.csv'
test_data=pd.read_csv(d_path+'/testData.tsv',header=0,delimiter='\t',quoting=3)
clean_test_reviews=[]
for review in test_data['review']:
    clean_test_reviews.append(preprocessing(review,remove_stopwords=True))
clean_test_df=pd.DataFrame({'review':clean_test_reviews,'id':test_data['id']})
tokenizer=Tokenizer()
tokenizer.fit_on_texts(clean_test_reviews)
text_sequences=tokenizer.texts_to_sequences(clean_test_reviews)
test_inputs=pad_sequences(text_sequences,maxlen=174,padding='post')
test_id=np.array(test_data['id'])

#저장부
np.save(open(d_path+test_input_data,'wb'),test_inputs)
np.save(open(d_path+test_id_data,'wb'),test_id)
clean_test_df.to_csv(d_path+test_clean_data,index=False)