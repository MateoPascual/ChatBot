import json
import random
import numpy
import nltk
nltk.download('punkt')

with open("Knowledge_Base.json") as json_data:
  data = json.load(json_data)


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

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
words_lowercase = [stemmer.stem(word.lower()) for word in words]
words = sorted(list(set(words_lowercase)))

#  Build bag of words for ML model



empty_output = [0] * len(classes)



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


"""#  Split data for machine learning"""

random.shuffle(training_data)
training_numpy = numpy.array(training_data)

print(training_numpy)

print(type(training_numpy))

train_X = list(training_numpy[:,0])

print(train_X)

print(len(train_X))

train_y = list(training_numpy[:,1])

print(train_y)
