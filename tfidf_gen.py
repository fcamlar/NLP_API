import nltk
import string
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')
import functools
import itertools
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import re

stopwords = nltk.corpus.stopwords.words('english')
ps = LancasterStemmer() 

def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation]) # Discount punctuation
    tokens = re.split('\W+', txt) # Split words into a list of strings
    txt = [ps.stem(word) for word in tokens if word not in stopwords] #Stem words
    return txt

def tfidf(query, paras):

    tfidf_vect = TfidfVectorizer(analyzer=clean_text)
    corpus = tfidf_vect.fit_transform(paras)
    query = tfidf_vect.transform([query])

    cosineSimilarities = cosine_similarity(corpus, query, dense_output = False)
    cos_df = cosineSimilarities.toarray()
    
    
    # Generate table of of top matches
    
    Match_percent = [i*100 for i in cos_df] # calculate percentage of match 
    matches = sorted([(x,i) for (i,x) in enumerate(Match_percent)], reverse=True)
    # index and percentage from cos_df
    idx = [item[1] for item in matches]
    
    matches = [item[0] for item in matches] # get the percentage
    matches = [int(float(x)) for x in matches] # convert to integer from np.array
    matches =  [i for i in matches if i > 0] # remove those lower than 20%
    #matches = [str(i) for i in matches] # convert int to string for percentage
    #matches = list(map("{}%".format, matches))
    
    # take first n elements of idx to mirror matches
    idx = idx[:len(matches)]

    ### Must list of lists to list of integers
    
    Paragraph = [paras[i] for i in idx] 
    
    result = pd.DataFrame({'Paragraph': Paragraph, 'Match Percentage': matches})

    return result