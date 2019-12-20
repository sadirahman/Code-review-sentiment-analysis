import collections
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import code_reviews



def word_feats(words):
    return dict([(word, True) for word in words])


efficient = code_reviews.fileids('Eff')
NotEff = code_reviews.fileids('NotEff')

efficientfeats = [(word_feats(code_reviews.words(fileids=[f])), 'neg') for f in negids]
NotEfffeats = [(word_feats(code_reviews.words(fileids=[f])), 'pos') for f in posids]

efficientcutoff = len(efficientfeats) * 3 // 4
NotEffcutoff = len(NotEfffeats) * 3 // 4

trainfeats = efficientfeats[:efficientcutoff] + NotEfffeats[:NotEffcutoff]
testfeats = efficientfeats[efficientcutoff:] + NotEfffeats[NotEffcutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print('Eff precision:', nltk.metrics.scores.precision(refsets['Eff'], testsets['Eff']))
print('Eff recall:', nltk.metrics.scores.recall(refsets['Eff'], testsets['Eff']))
print('Eff F-measure:', nltk.metrics.scores.f_measure(refsets['Eff'], testsets['Eff']))
print('NotEff precision:', nltk.metrics.scores.precision(refsets['NotEff'], testsets['NotEff']))
print('NotEff recall:', nltk.metrics.scores.recall(refsets['NotEff'], testsets['NotEff']))
print('NotEff F-measure:', nltk.metrics.scores.f_measure(refsets['NotEff'], testsets['NotEff']))
