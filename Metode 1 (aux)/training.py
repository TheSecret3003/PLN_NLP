#Import Library
import numpy as np
import pandas as pd
import re
import string
from sklearn.metrics import f1_score,accuracy_score
from simpletransformers.classification import ClassificationModel
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import warnings
warnings.simplefilter("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Read Data Training
train = pd.read_csv('DATASET/train_aux.csv')
val = pd.read_csv('DATASET/val_aux.csv')
test_real = pd.read_csv('DATASET/test_aux.csv')
test = test_real.copy()

def clean_text(text):
    text = text.lower()
    text = re.sub('[%s]' %re.escape(string.punctuation.replace('?', '')), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
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
train['text_a'] = train['text_a'].apply(clean_text)
val['text_a'] = val['text_a'].apply(clean_text)
test['text_a'] = test['text_a'].apply(clean_text)

# ##remove stopwords
# with open('DATASET/stopwords.txt') as f:
#         stopwords_list = f.read().splitlines()
# train['text_a'] = train['text_a'].apply(lambda x :remove_stopwords(x,stopwords_list))
# val['text_a'] = val['text_a'].apply(lambda x :remove_stopwords(x,stopwords_list))
# test['text_a'] = test['text_a'].apply(lambda x :remove_stopwords(x,stopwords_list))

#Logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_type", default="bert", help="model type")
parser.add_argument("--model", default='bert_models', help=" Base Model Folder")
parser.add_argument("--num_epoch" , default=1, help="number of train epoch")
parser.add_argument("--num_gpu" , default=1, help="number of gpu use")
parser.add_argument("--cuda_device" , default=2, help="cuda device")
argss = vars(parser.parse_args())

TRAIN_BATCH = 32
# Create a ClassificationModel
model = ClassificationModel(argss['model_type'], 'BASE_MODELS/' + argss['model'], use_cuda=True, cuda_device= int(argss['cuda_device']),
                            args={
    'reprocess_input_data': True,
    "learning_rate": 2e-5,
    "train_batch_size" : TRAIN_BATCH,
    "best_model_dir" : "Models/{}/bestModel".format(argss['model']),
    "output_dir" : "Models/checkpoints/{}".format(argss['model']),
    "evaluate_during_training" : True,
    "evaluate_during_training_steps" : int(np.ceil(train.shape[0]/TRAIN_BATCH)),
    'overwrite_output_dir': True,
    'num_train_epochs': int(argss['num_epoch']),    "save_eval_checkpoints": False, "save_model_every_epoch" : False, 
    "save_steps": -1,
    "use_multiprocessing": False, 
    "use_multiprocessing_for_evaluation": False, 
    "process_count": 1,
    "no_cache" : True,
    "n_gpu" : int(argss['num_gpu'])}
)

#Training
model.train_model(train, eval_df=val)

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

#save predictions to csv
test_real['prediksi'] = predictions

## f1 score normalization
Kalimat = []
aspek = []
labels = []
prediksi = []
for idx in test_real.index:
    if test_real['labels'][idx]== 1:
        Kalimat.append(test_real['text_a'][idx])
        a = str(test_real['text_b'][idx]).split('-')
        if len(a)==3:
            aspek.append(a[0]+' '+a[1])
        elif len(a) == 2:
            aspek.append(a[0])
        labels.append(test_real['labels'][idx])
        prediksi.append(test_real['prediksi'][idx])
df_new = pd.DataFrame()
df_new['kalimat'] = Kalimat
df_new['aspek'] = aspek
df_new['label'] = labels
df_new['pred'] = prediksi

labels_true = df_new['label'].values.tolist()
predictions = df_new['pred'].values.tolist()

#F1 Score
f1 = f1_score(labels_true, y_pred=predictions, average='micro')
acc = accuracy_score(labels_true, predictions)

# Print Score
print("F1 Score Best Model : {}".format(f1))
print('Accuracy Score Best Model : {}'.format(acc))
