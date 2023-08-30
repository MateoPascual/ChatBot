print('hello')

import json

with open("data.json") as json_data:
  data = json.load(json_data)

print(data)

import nltk

nltk.download('punkt')

words = []
documents = []
classes = []

for intent in data["intents"]:
  for pattern in intent["patterns"]:
    word = nltk.word_tokenize(pattern)

    words.extend(word)
    documents.append((word, intent["tag"]))

    if intent["tag"] not in classes:
      classes.append(intent["tag"])

print(words)

print(documents)

print(classes)


from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

words_lowercase = [stemmer.stem(word.lower()) for word in words]

print(words_lowercase)

print(set(words_lowercase))

words = sorted(list(set(words_lowercase)))

print(words)

"""#  Build bag of words for ML model"""

print(documents)

empty_output = [0] * len(classes)

print(empty_output)

training_data = []

for document in documents:
  bag_of_words = []

  pattern_words = document[0]
  pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

  for word in words:
    bag_of_words.append(1) if word in pattern_words else bag_of_words.append(0)

  output_row = list(empty_output)
  output_row[classes.index(document[1])] = 1
  training_data.append([bag_of_words, output_row])

print(pattern_words)

print(training_data)

"""#  Split data for machine learning"""

import random

random.shuffle(training_data)

print(training_data)

print(type(training_data))

import numpy

training_numpy = numpy.array(training_data)

print(training_numpy)

print(type(training_numpy))

train_X = list(training_numpy[:,0])

print(train_X)

print(len(train_X))

train_y = list(training_numpy[:,1])

print(train_y)

"""#  Build a TensorFlow machine learning model for chat"""

import tflearn

neural_network = tflearn.input_data(shape = [None, len(train_X[0])])

print(neural_network)

neural_network = tflearn.fully_connected(neural_network, 8)

print(neural_network)

neural_network = tflearn.fully_connected(neural_network, 8)

print(neural_network)

neural_network = tflearn.fully_connected(neural_network, len(train_y[0]), activation="softmax")

print(neural_network)

neural_network = tflearn.regression(neural_network)

print(neural_network)

model = tflearn.DNN(neural_network)

print(model)

model.fit(train_X, train_y, n_epoch = 2000, batch_size = 8, show_metric = True)

"""#  Test chatbot machine learning model"""

model.save("chatbot_dnn.tflearn")

model.load("chatbot_dnn.tflearn")

print(model)

question = "Do you sell any coding course?"

def process_question(question):
  question_tokenized = nltk.word_tokenize(question)

  question_stemmed = [stemmer.stem(word.lower()) for word in question_tokenized]

  bag = [0] * len(words)

  for stem in question_stemmed:
    for index, word in enumerate(words):
      if word == stem:
        bag[index] = 1

  return(numpy.array(bag))

process_question(question)

prediction = model.predict([process_question(question)])

print(prediction)

print(classes)

"""#  Categorize chat question with ML"""

def categorize(prediction):

  prediction_top = [[index, result] for index,result in enumerate(prediction) if result > 0.5]

  prediction_top.sort(key=lambda x: x[1], reverse = True)

  result = []

  for prediction_value in prediction_top:
    result.append((classes[prediction_value[0]], prediction_value[1]))

  return result

categorize(prediction[0])

def chatbot(question):

  prediction = model.predict([process_question(question)])

  result = categorize(prediction[0])

  return result

chatbot("Do you have non coding content?")

chatbot("What products do you have?")

"""#  Pick a chatbot response in top category"""

user_input = input("Do you have a question for me?")

print(user_input)

def respond_to_input(user_input):

  question_category = chatbot(user_input)

  if question_category:
    while question_category:
      for intent in data["intents"]:
        if intent["tag"] == question_category[0][0]:
          return random.choice(intent["responses"])

respond_to_input(user_input)

for i in range(4):
  user_input = input("Do you have a question for me?\n")
  response = respond_to_input(user_input)
  print(response)
