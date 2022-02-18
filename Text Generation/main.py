import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import string
import os
import urllib3.request


token_df = pd.read_csv('token_df.csv')

to_token = token_df['text_clean'][0]+token_df['text_clean'][1]+token_df['text_clean'][2]+token_df['text_clean'][3]+token_df['text_clean'][4]
to_token.replace('\n',' ')
texts = re.split(r'[.,]', to_token)

def remove_punctuation(input_):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(input_)
    text = " ".join(text)
    return text

final_text=[]
for i in texts:
    final_text.append(remove_punctuation(i))

def create_character_tokenizer(list_of_strings):
    tokenizer = Tokenizer(filters=None,char_level=True, split=None, lower=False)
    tokenizer.fit_on_texts(list_of_strings)
    return tokenizer

tokenizer = create_character_tokenizer(final_text)


def get_model(vocab_size, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim = 256, mask_zero=True, batch_input_shape=(batch_size, None)),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=512, return_sequences=True,stateful=True),
        tf.keras.layers.Dense(units=vocab_size)
    ])
    return model


model = get_model(91, batch_size=1)
model.load_weights(tf.train.latest_checkpoint('./models/')).expect_partial()

def get_logits(model, token_sequence, initial_state1=None):
    
    token_sequence = np.asarray(token_sequence)
    if initial_state1 is not None:
        model.layers[1].states = initial_state1
    else:
        model.layers[1].reset_states()
    logit = model.predict(token_sequence)
    logit = logit[:,-1,:]
    
    return logit

def sample_token(logits):
    pred = tf.random.categorical(logits, num_samples=1).numpy()[0]
    return pred[0]

input_word = input("Input Word ")
no_of_words = input("Input Number of words ")

init_string = input_word
num_generation_steps = int(no_of_words)

token_sequence = tokenizer.texts_to_sequences([init_string])
initial_state_1, initial_state_2 = None, None
input_sequence = token_sequence

for _ in range(num_generation_steps):
    logits = get_logits(model, 
                        input_sequence, 
                        initial_state1=initial_state_1)
    sampled_token = sample_token(logits)
    token_sequence[0].append(sampled_token)
    input_sequence = [[sampled_token]]
    initial_state_1 = model.layers[1].states
    
print(tokenizer.sequences_to_texts(token_sequence)[0][::2])
