from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("vehicle",pos="a"))
print(lemmatizer.lemmatize("good",pos="a"))
print(lemmatizer.lemmatize("run",pos="a"))