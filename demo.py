import pandas as pd
import random, sys
import os
import re
import string
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax

aspect = ['Transisi enrgi','Teknologi Informasi','Budaya Green','Pegawai','Keselarasan Strategi','Tata Kelola','Metode']

def generate_sentence_pair(text):
  sentence_pairs = []
  sentence_pair1 = []
  sentence_pair2 = []
  aspect_sentiment = []
  for i in aspect:
      pair1 = i+"-pos"
      pair2 = i+"-neg"
      sentence_pair1.append(text)
      sentence_pair1.append(pair1)
      sentence_pair2.append(text)
      sentence_pair2.append(pair2)
      sentence_pairs.append(sentence_pair1)
      sentence_pairs.append(sentence_pair2)
      aspect_sentiment.append(pair1)
      aspect_sentiment.append(pair2)
      sentence_pair1 = []
      sentence_pair2 = []
  return sentence_pairs, aspect_sentiment

def clean_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation.replace('?', '')), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\r', '', text)
    text = text.replace('?', ' ?')
    text = text.replace('\d+', '')
    text = re.sub('[.;:!\'?,\"()\[\]*~]', '', text)
    text = re.sub('(<br\s*/><br\s*/>)|(\-)|(\/)', '', text)
    return text

bert_model = ClassificationModel('bert', 'bert_model',use_cuda=False)

text = "Saya bisa mengakses banyak ilmu tentang tindakan ramah lingkungan"
text = clean_text(text)
def make_prediction(text):
  sentence_pairs, aspect_sentiment = generate_sentence_pair(text)
  predictions, raw_outputs = bert_model.predict(sentence_pairs)
  test = pd.DataFrame(columns=["aspect-sentiment","label"])
  probs = softmax(raw_outputs,axis=1)
  prb = []
  for pr in probs:
      prb.append(max(pr[0],pr[1]))
  test['peluang'] = prb
  test['aspect-sentiment'] = aspect_sentiment
  test['label'] = predictions
  result = test[test['label'] == 1]
  pattern = 'pos|neg'
  results = result['aspect-sentiment'].str.contains(pattern)
  print(result)

make_prediction(text)