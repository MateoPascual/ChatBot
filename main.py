import json
import nltk
from nltk.stem.lancaster import  LancasterStemmer
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


#cleaning data with stemmer
stemmer = LancasterStemmer()
words = [stemmer.stem(word.lower()) for word in words]
words = sorted(list(set(words)))

#Bag of words

traing_data = []
empty_output = [0] * len(classes)

for document in documents:
  bag_of_words = []
  pattern_words = document[0]
  pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

  for word in words:
    bag_of_words.append(1) if word in pattern_words else bag_of_words.append(0)

output_row = list(empty_output)
output_row[classes.index(document[1])] = 1
traing_data.append([bag_of_words,output_row])

