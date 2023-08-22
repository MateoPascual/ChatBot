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

print(words)
