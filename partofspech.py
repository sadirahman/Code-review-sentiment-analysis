import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = "i love you "
sample_text = "i hate you"

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process():

    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged =nltk.pos_tag(words)
            print(tagged)
    except exception as a:
        print(str(a))

process()


