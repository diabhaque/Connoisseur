import base64
import io
from PIL import Image
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
import json
import re
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models
import pickle

# Defining the model (to be moved to seperate file)

# Feel free to change these parameters according to your system's configuration

top_k = 5000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))



# Defining the server
app=Flask(__name__)
CORS(app)

global image_features_extract_model
global encoder
global tokenizer 
global decoder

image_features_extract_model = tf.keras.models.load_model('./image_features_extract_model')

encoder=tf.keras.models.load_model('./encoder')

decoder = RNN_Decoder(embedding_dim, units, vocab_size)

decoder.load_weights('./save/my_save')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# encoder.summary()

print("Model loaded")

print("app is running...")

@app.route("/predict", methods=["POST"])


def predict():
    message=request.get_json(force=True)
    # Get encoded image in base 64
    encoded=message['image']
    # Remove file headers
    encoded=re.sub('^data:image/.+;base64,', '', encoded)
    #  Decode to bytes
    decoded=base64.b64decode(encoded)
    # Read bytes
    bytesIOimage=io.BytesIO(decoded)
    # Covert to PIL Image
    image=Image.open(bytesIOimage)
    # Converr to NumPy array
    img= np.asarray(image)/255
    # print(img)
    # Convert to tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    # img = tf.image.decode_jpeg(img, channels=3)
    # Resize tensor

    img = tf.image.resize(img, (299, 299))

    img=tf.expand_dims(img, 0)

    hidden = decoder.reset_state(batch_size=1)

    img_tensor_val = image_features_extract_model(img)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    beam_index=3
    
    sequences=[[[], 0.0]]

    for i in range(50):
        all_candidates=list()
        for s in sequences:
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            # word_preds=np.argsort(predictions[0])[-beam_index:]

            # for w in word_preds:
            #     next_cap, prob=s[0][:], s[1]
            #     next_cap.append(w)
            #     prob += preds[0][w]
            #     temp.append([next_cap, prob])

            # if(i<2):
            #     print(temp)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            
            if tokenizer.index_word[predicted_id] == '<end>':
                break
            
            result.append(tokenizer.index_word[predicted_id])

            dec_input = tf.expand_dims([predicted_id], 0)

    response={
        "res": ' '.join(result),
    }

    return jsonify(response)