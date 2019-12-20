from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def pre_proces():
    lemmatizer = WordNetLemmatizer()
    example_sentance = "This is example showing off stop word filtration."
    stop_words = set(stopwords.words('english'))
    word = word_tokenize(example_sentance)
    filtered_sentance = []
    for i in word:
        if i not in stop_words:
            filtered_sentance.append(i)
    print(filtered_sentance)


pre_proces()