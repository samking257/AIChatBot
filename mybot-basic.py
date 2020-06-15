#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, requests, urllib.parse, itertools

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

#for bag of words model etc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # faster than cosine similarity

import xml.etree.ElementTree as et
import nltk

import tensorflow_datasets as tfds
import gym
import random
from collections import deque

env_name = "CartPole-v0"
env = gym.make(env_name)

#  Initialise AIML agent
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel.
kern.bootstrap(learnFiles="mybot-basic.xml")

#   Initialise first order logic based agent
v = """
vodka => {}
rum => {}
gin => {}
juice => {}
lime => {}
strawberries => {}
coke => {}
apples => {}
whiskey => {}
lemonade => {}
coffee => {}
orange => {}
tonic => {}
lemon => {}

recipe0 => r0
recipe1 => r1
recipe2 => r2
recipe3 => r3
recipe4 => r4
recipe5 => r5
recipe6 => r6
recipe7 => r7
recipe8 => r8
recipe9 => r9

be_in => {}
"""

folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0

LabelsList = ["Apple", "Banana", "Lemon", "Lime", "Orange", "Strawberry", "Tomato"]
dir_path = os.path.dirname(os.path.realpath(__file__))
#model = tf.keras.models.load_model(dir_path + "\\CNNfruits.h5")

recipeList = []

# Warn User - Ethics
print("*** Warning: Please drink responsibly ***")
# Welcome user
print("Welcome, my name is Ms Bartender and I am here to help you become a cocktail master!",
      "I can provide many cocktail recipes or you can create your own. Happy mxing! :)")

def IsIngredInRecipe(ingred, recipe):
    if ('',) not in folval["be_in"] and ('',) not in folval[ingred]:
        item = folval[ingred]
        numlist = []
        for element in item:
            numlist.append(element[0])
        beIn = folval["be_in"]
        for i in beIn:
            if i[1] == folval[recipe]:
                if i[0] in numlist:
                    return True
    return False

#Transformer Network
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_en.vocab_size + 2
target_vocab_size = tokenizer_pt.vocab_size + 2
dropout_rate = 0.1

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = (dir_path + "\\" + "Translation")

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)

MAX_LENGTH = 40
def evaluate(inp_sentence):
  start_token = [tokenizer_en.vocab_size]
  end_token = [tokenizer_en.vocab_size + 1]
  
  # inp sentence is english, hence adding the start and end token
  inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is portiguese, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_pt.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
        output, False, enc_padding_mask, combined_mask, dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_pt.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

def translate(sentence):
  result, attention_weights = evaluate(sentence)
  
  predicted_sentence = tokenizer_pt.decode([i for i in result 
                                            if i < tokenizer_pt.vocab_size])  

  print('Input: {}'.format(sentence))
  print('Bot: {}'.format(predicted_sentence))

# Neural Network
class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
        
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        
    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)
        
    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))

class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.replay_buffer = ReplayBuffer(maxlen=10000)
        self.gamma = 0.97
        self.eps = 1.0
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action
    
    def train(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(50)
        q_next_states = self.q_network.get_q_state(self.sess, next_states)
        q_next_states[dones] = np.zeros([self.action_size])
        q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)
        self.q_network.update_model(self.sess, states, actions, q_targets)
        
        if done: self.eps = max(0.1, 0.99*self.eps)
    
    def __del__(self):
        self.sess.close()

# Main loop
while True:
    #get user input
    try:
        userInput = input("You: ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print("Bot:",params[1])
            break
        elif cmd == 1:
            succeeded = False
            cocktail_api = r"https://www.thecocktaildb.com/api/json/v1/1/search.php?s"
            url = cocktail_api + urllib.parse.urlencode({ '' : params[1]}) #add cocktail to search
            response = requests.get(url)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json['drinks'] != None:
                    name = response_json['drinks'][0]['strDrink']
                    instructions = response_json['drinks'][0]['strInstructions']
                    #loop through ingredients
                    ingredients = response_json['drinks'][0]['strIngredient1']
                    ingredientsList = []
                    increment = 1
                    while ingredients != None :
                        ingredientsList.append(ingredients)
                        increment = increment + 1
                        ingredients = response_json['drinks'][0]['strIngredient'+ str(increment)]
                    #loop through measurements
                    measurements = response_json['drinks'][0]['strMeasure1']
                    measurementsList = []
                    increment = 1
                    while measurements != None :
                        measurementsList.append(measurements)
                        increment = increment + 1
                        measurements = response_json['drinks'][0]['strMeasure'+ str(increment)]
                    #put lists together
                    ingredMeasure = itertools.zip_longest(measurementsList,ingredientsList, fillvalue=None)
                    #print answers
                    print("Bot: Got it! Here is your recipe:")
                    print("Cocktail:", name)
                    print("Ingredient: ")
                    for x in tuple(ingredMeasure): 
                        if x[0] != None:
                            print(x[0], x[1])
                        else: print(x[1])
                    print("Instructions: ", instructions)
                    succeeded = True
            if not succeeded:
                print("Bot: Sorry, I could not find the cocktail you gave me.")
        elif cmd == 2:
            name = params[1].split()
            picture_name = (name[0] + "." + params[2])
            image_path = (dir_path + "\\" + picture_name)

            def prepare(filepath):
                img_size = 100
                img_array = cv2.imread(filepath)
                new_array = cv2.resize(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), (img_size, img_size))
                new_array = new_array/255
                return new_array.reshape(-1, img_size, img_size, 3)

            def displayImage(filepath):
                img_size = 100
                img_array = cv2.imread(filepath)
                new_array = cv2.resize(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), (img_size, img_size))
                new_array = new_array/255
                plt.imshow(new_array)
                plt.show()
            try:
                open(image_path)
            except:
                print("Bot: I could not find the image.")
            else:
                prediction = model.predict_classes([prepare(image_path)])
                print("Bot: Lemme look ... " + LabelsList[int(prediction)] + "!!")
                displayImage(image_path)
        elif cmd == 3: # change name
            if len(recipeList) == 10:
                print("Bot: No more recipes availiable")
            else:
                if params[1] not in recipeList:
                    recipeList.append(params[1])
                    print("Bot: New recipe created, you can now add ingredients to it.")
                else:
                    print("Bot: Recipe name already exists")
        elif cmd == 4: # I will add ingredient x in recipe y
            if folval.__contains__(params[1]) == False:
                print("Bot: Sorry but the ingredient does not exist")
            else:
                if params[2] not in recipeList:
                    print("Bot: The recipe does not exist")
                else:
                    recipe = "recipe" + str(recipeList.index(params[2]))
                    if IsIngredInRecipe(params[1], recipe) == True:
                        print("Bot: Ingredient already in recipe")
                    else:
                        o = 'o' + str(objectCounter)
                        objectCounter += 1
                        folval['o' + o] = o #insert constant
                        if len(folval[params[1]]) == 1: #clean up if necessary
                            if ('',) in folval[params[1]]:
                                folval[params[1]].clear()
                        folval[params[1]].add((o,)) #insert type of ingredient information
                        if len(folval["be_in"]) == 1: #clean up if necessary
                            if ('',) in folval["be_in"]:
                                folval["be_in"].clear()
                        folval["be_in"].add((o, folval[recipe])) #insert location
                        print("Bot: Added!")
        elif cmd == 5: #Are there any x in y
            if folval.__contains__(params[1]) == False:
                print("Bot: Sorry but the ingredient does not exist")
            else:
                if params[2] not in recipeList:
                    print("Bot: The recipe does not exist")
                else:
                    g = nltk.Assignment(folval.domain)
                    m = nltk.Model(folval.domain, folval)
                    recipe = "recipe" + str(recipeList.index(params[2]))
                    sent = 'some ' + params[1] + ' are_in ' + recipe
                    results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                    if results[2] == True:
                        print("Yes.")
                    else:
                        print("No.")
        elif cmd == 6: # Are all x in y
            if folval.__contains__(params[1]) == False:
                print("Bot: Sorry but the ingredient does not exist")
            else:
                if params[2] not in recipeList:
                    print("Bot: The recipe does not exist")
                else:
                    g = nltk.Assignment(folval.domain)
                    m = nltk.Model(folval.domain, folval)
                    recipe = "recipe" + str(recipeList.index(params[2]))
                    sent = 'all ' + params[1] + ' are_in ' + recipe
                    results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                    if results[2] == True:
                        print("Bot: Yes")
                    else:
                        print("Bot: No")
        elif cmd == 7: # Which ingredient are in ...
            if params[1] not in recipeList:
                print("Bot: Sorry but the recipe does not exist")
            else:
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                recipe = "recipe" + str(recipeList.index(params[1]))
                e = nltk.Expression.fromstring("be_in(x," + recipe + ")")
                sat = m.satisfiers(e, "x", g)
                if len(sat) == 0:
                    print("Bot: Nothing")
                else:
                    print("Bot: " + params[1] + "'s Ingredients are:")
                    #find satisfying objects in the valuation dictionary,
                    #and print their type names
                    sol = folval.values()
                    for so in sat:
                        for k, v in folval.items():
                            if len(v) > 0:
                                vl = list(v)
                                if len(vl[0]) == 1:
                                    for i in vl:
                                        if i[0] == so:
                                            print(k)
                                            break
        elif cmd == 8: # I will remove ingredient x in recipe y
            if folval.__contains__(params[1]) == False:
                print("Bot: Sorry but the ingredient does not exist")
            else:
                if params[2] not in recipeList:
                    print("Bot: The recipe does not exist")
                else:
                    recipe = "recipe" + str(recipeList.index(params[2]))
                    objectNumber = ""
                    if ('',) not in folval["be_in"] and ('',) not in folval[params[1]]:
                        item = folval[params[1]]
                        numlist = []
                        for element in item:
                            numlist.append(element[0])
                        beIn = folval["be_in"]
                        for i in beIn:
                            if i[1] == folval[recipe]:
                                if i[0] in numlist:
                                    objectNumber = i[0]
                        folval["be_in"].remove((objectNumber, folval[recipe])) #remove from recipe
                        folval[params[1]].remove((objectNumber,)) # remove objectnumber
                        print("Bot: Removed!")
                    else:
                        print("Bot: Ingredient is not in this recipe")
        elif cmd == 9:
            translate(params[1])
        elif cmd == 10:
            def is_number(num):
                try:
                    float(num)
                    return True
                except ValueError:
                    return False

            if(is_number(params[1])):
                num_episodes = (int)params[1]
                if(num_episodes > 0 & num_episodes < 1001):
                    agent = DQNAgent(env)
                    best_episode = 0
                    best_reward = 0
                    for ep in range(num_episodes):
                        state = env.reset()
                        total_reward = 0
                        done = False
                        while not done:
                            action = agent.get_action(state)
                            next_state, reward, done, info = env.step(action)
                            agent.train(state, action, next_state, reward, done)
                            env.render()
                            total_reward += reward
                            state = next_state
                        if(total_reward > best_reward):
                            best_reward = total_reward
                            best_episode = total_reward    
                        #print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))
                    print("Bot: Best episode was " + best_episode + " with the reward of " + best_reward)
                else:
                    print("Bot: Invalid input.  Please enter a number between 1 and 1000.")
            else:
                print("Bot: Invalid input. Please enter a number.")
        elif cmd == 99:
            #parse xml
            root = et.parse('mybot-basic.xml').getroot()
            aimlList = [] 
            #Loop through
            for i in root:
                find = i.find('pattern')
                if "*" not in find.text:
                    aimlList.append(find.text)
            
            aimlList.insert(0, params[1]) #insert user's input

            tfidf = TfidfVectorizer().fit_transform(aimlList) #TF-IDF
            cosineSimilarities = linear_kernel(tfidf[0:1], tfidf).flatten() #Cosine similarity
            related_docs = cosineSimilarities.argsort()[:-3:-1] #most similar
            most_similar = related_docs[1] #doc most similar to query 0

            #get response
            if cosineSimilarities[most_similar] > 0.3:
                answer = kern.respond(aimlList[most_similar])
                print("Bot:", answer)
            else:
                print("Bot: Sorry, I did not get that. Please try again.")
    else:
        print("Bot:", answer)