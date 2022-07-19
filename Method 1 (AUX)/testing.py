#Import Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,classification_report, accuracy_score
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import os.path
import sys
import warnings
warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

test = pd.read_csv('DATASET/test_aux.csv')
test_real = test.copy()

def clean_text(text):
    import re
    import string
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

#remove punctuation (clean text)
test['text_a'] = test['text_a'].apply(clean_text)

# ##remove stopwords
# with open('DATASET/stopwords.txt') as f:
#         stopwords_list = f.read().splitlines()
# test['text_a'] = test['text_a'].apply(lambda x :remove_stopwords(x,stopwords_list))   

#Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_type", default="bert", help="model type")
parser.add_argument("--model", default='bert_models', help=" Base Model Folder")
parser.add_argument("--num_epoch" , default=1, help="number of train epoch")
argss = vars(parser.parse_args())

model = ClassificationModel(argss['model_type'], "Models/{}/bestModel".format(argss['model']), use_cuda=True, cuda_device=1, 
        args={"use_multiprocessing": False, 
              "use_multiprocessing_for_evaluation": False, 
              "process_count": 1}) 
##Prediction
string = []
strings = []
for i in test.index:
  string.append(test['text_a'][i])
  string.append(test['text_b'][i])
  strings.append(string)
  string = []
predictions, raw_outputs = model.predict(strings)

probs = softmax(raw_outputs,axis=1)

#save predictions to csv
test_real['prediksi'] = predictions

## f1 score normalization
Kalimat = []
aspek = []
labels = []
prediksi = []
sentimen = []
for idx in test_real.index:
    if test_real['labels'][idx]== 1:
        Kalimat.append(test_real['text_a'][idx])
        a = str(test_real['text_b'][idx]).split('-')
        if len(a)==3:
            aspek.append(a[0]+' '+a[1])
            sentimen.append(a[2])
        elif len(a) == 2:
            aspek.append(a[0])
            sentimen.append(a[1])
        labels.append(test_real['labels'][idx])
        prediksi.append(test_real['prediksi'][idx])

df_new = pd.DataFrame()
df_new['kalimat'] = Kalimat
df_new['aspek'] = aspek
df_new['sentimen'] = sentimen
df_new['label'] = labels
df_new['pred'] = prediksi

# df_new['kalimat'] = df_new['kalimat'].drop_duplicates(keep='first')
# df_new = df_new[df_new['kalimat'].notnull()]

labels_true = df_new['label'].values.tolist()
predictions = df_new['pred'].values.tolist()

#F1 Score
f1 = f1_score(labels_true, y_pred=predictions, average='micro')
acc = accuracy_score(labels_true, predictions)

#Save result (prediction and F1 score)
script_path = ""
new_abs_path = os.path.join(script_path, f"Hasil/{argss['model']}")
if not os.path.exists(new_abs_path):
  os.mkdir(new_abs_path)

df_new.to_csv('Hasil/{}/prediksi_{}.csv'.format(argss['model'], argss['model']))
with open('Hasil/{}/score_F1_{}.txt'.format(argss['model'], argss['model']), 'w') as f:
    f.write('F1 Score : {}'.format(f1))
    f.write('\n')
    f.write('Accuracy Score : {}'.format(acc))

# Print in txt format
print("F1 Score Best Model : {}".format(f1))
print('Accuracy Score Best Model : {}'.format(acc))




