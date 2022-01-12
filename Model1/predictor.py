import json
import numpy as np
import pandas as pd
from itertools import islice
from more_itertools import chunked
from emotion_classifier.predictor import EmotionPredictor

clr = EmotionPredictor.from_path(archive_path='models/emotion_classifier_2/model.tar.gz', predictor_name='emotion_predictor', cuda_device=0)
labels = {
    0: 'anger', 
    1: 'anticipation', 
    2: 'disgust', 
    3: 'fear', 
    4: 'sadness', 
    5: 'surprise', 
    6: 'trust', 
    7: 'joy'
}


def preprocess(line):
    return {'text': line}


INPUT_FILE = ""
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)


text_data = iter(data['text'])
text_data_list = [ text_dict for text_dict in map(preprocess, text_data) ]

text_data_iter = iter(text_data_list)

text_list = []
predicts_list = []

while (batch_lines := tuple(islice(text_data_iter, 30))):
    probs_list = clr.predict_batch_json(batch_lines)
    for result in probs_list:
        text_list.append(result['text'])
        predicts_list.append(labels[np.argmax(result['probs'])])

SAMPLE_FILE, TEST_DATA,  = "", ""
sample = pd.read_csv(SAMPLE_FILE)
test_data_df = pd.read_pickle(TEST_DATA)
output = pd.merge(sample, test_data_df, left_on='id', right_on='tweet_id')
output = output.drop(columns=['emotion'])

answer_df = pd.DataFrame({
    'text': text_list,
    'emotion': predicts_list,
})

output = pd.merge(output, answer_df, on='text')

OUTPUT_FILE = ""
output.to_csv(OUTPUT_FILE, columns=['id', 'emotion'], index=False)