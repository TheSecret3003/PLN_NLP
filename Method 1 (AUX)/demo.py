#Import Library
from itertools import count
from pickle import FALSE
import numpy as np
import pandas as pd
import re
import string
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt')
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sqlalchemy import true
warnings.simplefilter("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

aspect = ['transisi-energi','teknologi-informasi','budaya-green','pegawai','keselarasan-strategi','tata-kelola','metode']

def generate_sentence_pair(text):
  sentence_pairs = []
  sentence_pair1 = []
  sentence_pair2 = []
  sentence_pair3 = []
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
      pair3 = i+"-none"
      sentence_pair3.append(text)
      sentence_pair3.append(pair3)
      sentence_pairs.append(sentence_pair3)
      aspect_sentiment.append(pair3)
      sentence_pair3 = []
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

# def remove_stopwords(teks, readline):
#    import re
#    for stopword in readline:
#         if stopword in teks:
#             rx = re.compile(r"\b{}\b".format(stopword))
#             teks = rx.sub("", teks)
#    return teks

#Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_type", default="bert", help="model type")
parser.add_argument("--model", default='bert_models', help=" Base Model Folder")
argss = vars(parser.parse_args())
model = ClassificationModel(argss['model_type'], "Models/{}/bestModel".format(argss['model']), use_cuda=True, cuda_device=1,
        args={"use_multiprocessing": False, 
              "use_multiprocessing_for_evaluation": False, 
              "process_count": 1}) 

while True:
    try:
        inp = input("Masukkan contoh teks : ")
        if type(inp) == str:
            break
    except ValueError:
        inp = input("input salah, silahkan masukkan contoh teks")

text = inp
# with open('DATASET/stopwords.txt') as f:
#         stopwords_list = f.read().splitlines()
# text = remove_stopwords(text, stopwords_list)

def make_prediction(text):
  sentence_pairs, aspect_sentiment = generate_sentence_pair(text)
  predictions, raw_outputs = model.predict(sentence_pairs)
  test = pd.DataFrame(columns=["aspect-sentiment","peluang"])
  test_result = pd.DataFrame(columns=["aspect-sentiment","peluang"])
  probs = softmax(raw_outputs,axis=1)
  prb = []
  preds = []
  for pr,pred in zip(probs,predictions):
      prb.append(pr[1])
      preds.append(pred)
  test['peluang'] = prb
  test['aspect-sentiment'] = aspect_sentiment
  test['pred'] = preds
  pattern = '-pos'
  results = test['aspect-sentiment'].str.contains(pattern)
  r = []
  p = []
  pred = []
  for i in test[results].index:
    result = test[results]['aspect-sentiment'][i].replace(pattern,'')
    r.append(result)
    p.append(test[results]['peluang'][i])
    pred.append(test[results]['pred'][i])
  test_result['aspect-sentiment'] = r
  test_result['peluang'] = p
  test_result['pred'] = pred
  test_result = test_result.sort_values(by=['peluang'], ascending=False)
  output = test_result.to_json(orient = 'records')

  return test_result

results = []
for teks in sent_tokenize(text) : 
  teks = clean_text(teks)
  result  = make_prediction(teks)
  result = result.sort_values(by='aspect-sentiment')
  results.append(result)

##Final probability
final_result = pd.DataFrame()
final_result['aspek'] = results[0]['aspect-sentiment']
final_result['peluang'] = 0

TRESHOLD = 0.2# probabily treshold
for idx in final_result.index:
  counter = 0
  if len(results) > 1:
    for tabel in results:
      if tabel['peluang'][idx] >= TRESHOLD:
        counter = counter + 1
      if tabel['peluang'][idx] < TRESHOLD:
        tabel.loc[idx,'peluang'] = 0
      final_result.loc[idx,'peluang']= final_result['peluang'][idx] + tabel['peluang'][idx]
    if counter !=0 : 
      final_result['peluang'][idx] = final_result['peluang'][idx]/counter
  else:
    final_result.loc[:,'peluang'] = results[0]['peluang']
final_result = final_result.sort_values(by='peluang', ascending=False)
print(final_result)
