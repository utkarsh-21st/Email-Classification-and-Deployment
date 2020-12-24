import tensorflow as tf
from app import app_dir, temp_dir, result_dir
import os
import extract_msg
import re
import pickle
import numpy as np
from transformers import DistilBertConfig, DistilBertTokenizerFast
import pandas as pd
import shutil

# patterns to be removed
pattern1 = re.compile(r'From: .*')
pattern2 = re.compile(r'Sent: .*')
pattern3 = re.compile(r'To: .*')
pattern4 = re.compile(r'[\n\r]')
pattern5 = re.compile(r'\d+')
pattern6 = re.compile(r'\S+@\S+')
pattern7 = re.compile(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]')
patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7]
pattern_a = re.compile(r'Cc: (.*)')


def load_tokenizer(num_classes):
    print('load tokenizer')
    # global tokenizer
    # global max_length
    # Name of the BERT model to use
    model_name = 'distilbert-base-uncased'
    max_length = pickle.load(open(app_dir / 'data' / 'model_data' / 'max_length.pickle', 'rb'))
    max_length = max_length
    # Load transformers config and set output_hidden_states to False
    config = DistilBertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    config.num_labels = num_classes

    # Load BERT tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name,
                                                        config=config)
    tokenizer = tokenizer
    return tokenizer, max_length


def load_model():
    print('load model')
    file_name = os.listdir(app_dir / 'model')[0]
    model_path = app_dir / 'model' / file_name
    model = tf.keras.models.load_model(model_path)
    return model


def tokenize(text, max_length, tokenizer):
    tokenized = tokenizer(text,
                          max_length=max_length,
                          padding=True,
                          truncation=True,
                          return_tensors='tf',
                          return_token_type_ids=False,
                          return_attention_mask=False,
                          verbose=True)
    return tokenized


def load_labelencoder():
    le = pickle.load(open(app_dir / 'data' / 'model_data' / 'le.pickle', 'rb'))
    num_classes = len(le.classes_)
    le = le
    num_classes = num_classes
    return le, num_classes


def remove_pattern(*patterns, text, group=0):
    '''
    :param patterns: list of patterns where each patter is a re.compile
    :param text: string, text
    :return: text, string, having all the patterns removed
    '''
    for pattern in patterns:
        matches = pattern.finditer(text)
        temp = ''
        pos_prev = 0
        pos_cur = 0
        for match in matches:
            pos_cur = match.span(0)[0]
            temp += text[pos_prev: pos_cur]
            temp += ' '
            pos_prev = match.span(0)[1]
        temp += text[pos_prev:]
        if temp != '':
            text = temp
    return text


def load_data():
    global data
    if len(os.listdir(temp_dir / 'messages')) == 0:
        print('No messages found')
        return
    # zip_name = os.listdir(temp_dir / 'messages')[0]
    # with zipfile.ZipFile(temp_dir / 'messages' / zip_name, 'r') as zip_ref:
    #     zip_ref.extractall(temp_dir / 'messages')
    # shutil.rmtree(temp_dir / 'messages' / zip_name)
    file_names = os.listdir(temp_dir / 'messages')
    data = pd.DataFrame(columns=['file_name', 'message'], index=range(len(file_names)))
    for i, file_name in enumerate(file_names):
        file_path = temp_dir / 'messages' / file_name
        msg = extract_msg.Message(file_path)
        msg_message = msg.body
        msg_message = remove_pattern(*[pattern_a], text=msg_message, group=1)
        msg_message = remove_pattern(*patterns, text=msg_message)
        data.iloc[i] = file_name, msg_message


def predict(max_length, le, model, tokenizer):
    categories = le.classes_
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for category in categories:
        if os.path.exists(result_dir / category):
            shutil.rmtree(result_dir / category)
        os.mkdir(result_dir / category)
    x_enc = tokenize(list(data.iloc[:, 1]), max_length=max_length, tokenizer=tokenizer)
    inp_ids = np.asarray(x_enc['input_ids'])
    # att_masks = np.asarray(x_enc['attention_mask'])
    # preds = np.empty(inp_ids.shape[0], dtype=int)
    preds = pd.DataFrame(columns=['file_name', 'category'], index=range(data.shape[0]))
    for i in range(inp_ids.shape[0]):
        inp_id = inp_ids[i: i + 1, :]
        # att_mask = att_masks[i: i+1, :]
        # pred = model(inputs=inp_id, attention_mask=att_mask)
        pred = model(inputs=inp_id)[0]
        pred = tf.math.argmax(pred)
        pred = int(pred.numpy())
        category = categories[pred]
        file_name = data.iloc[i, 0]
        preds.iloc[i] = file_name, category
        shutil.move(temp_dir / 'messages' / file_name, result_dir / category / file_name)
    preds.to_csv(result_dir / 'classification_result.csv', index=False)
    print(preds)
    print('classification done!')
    return ('classification done!')
