from sklearn.feature_extraction.text import CountVectorizer 
from nltk.tokenize import word_tokenize
# import nltk 
# nltk.download('punkt_tab')
 
corpus = [ 
    "hello, how are you?", 
    "im getting bored at home. And you? What do you think?", 
    "did you know about counts", 
    "let's see if this works!", 
    "YES!!!!" 
] 
ctv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None) 
ctv.fit(corpus) 
 
corpus_transformed = ctv.transform(corpus) 
print(ctv.vocabulary_)