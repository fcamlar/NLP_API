import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import functools
import itertools
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity


stopwords = nltk.corpus.stopwords.words('english')
ps = LancasterStemmer() 


cols = ['Title','Description','Urls']

data = pd.read_excel("links.xlsx", usecols = cols)
data["Title"] = data["Title"].astype(str)
data["Description"] = data["Description"].astype(str)
data['Links_TitleDesc'] = data[['Title', 'Description']].agg(' '.join, axis = 1)


def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation]) # Discount punctuation
    tokens = re.split('\W+', txt) # Split words into a list of strings
    txt = [ps.stem(word) for word in tokens if word not in stopwords] #Stem words
    return txt


def links(query):

    tfidf_vect = TfidfVectorizer(analyzer=clean_text)
    corpus = tfidf_vect.fit_transform(data['Links_TitleDesc'])
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
    matches =  [i for i in matches if i >= 20] # remove those lower than 20%
    matches = [str(i) for i in matches] # convert int to string for percentage
    matches = list(map("{}%".format, matches))
    
    # take first n elements of idx to mirror matches
    idx = idx[:len(matches)]

    ### Must list of lists to list of integers

    Title = [data.loc[i, 'Title'] for i in idx]
    Description = [data.loc[i, 'Description'] for i in idx]
    Url = [data.loc[i, 'Urls'] for i in idx] # Description of CD & KPI  

    results = pd.DataFrame({'Title': Title, 'Description':Description, 'Url':Url, 'Match Percentage': matches})
    results = results.to_dict()
    result= {
        "model": "links_search",
        "results": results
    }
    return result