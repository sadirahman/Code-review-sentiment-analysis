from nltk.tokenize import sent_tokenize ,word_tokenize

example_text = "hellow there how are you doing today?the weather is greate."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)
