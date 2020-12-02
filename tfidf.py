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


cols = ['Default Goal ID','Default Goal Description', 'Default Goal Name', 'Linked Initiatives IDs', 'Value Focus']

data = pd.read_excel("Goal_Data.xlsx", usecols = cols)
data["Default Goal Name"] = data["Default Goal Name"].astype(str)
data["Description"] = data["Default Goal Description"].astype(str)
data['DG_NameDesc'] = data[['Default Goal Description', 'Default Goal Name']].agg(' '.join, axis = 1)

data2 = pd.read_excel("Goal_Data.xlsx", sheet_name='Initiatives')
data2['I_NameDesc'] = data2[['Description', 'Initiative Name']].agg(' '.join, axis = 1)
data2 = data2.fillna(0)

data3 = pd.read_excel("Goal_Data.xlsx", sheet_name='Assets')
data3["Asset ID"] = data3["Asset ID"].str.replace(" ", "") # Remove spaces
data3["Asset Name"] = data3["Asset Name"].astype(str)
data3["Description"] = data3["Description, High"].astype(str)
data3['A_NameDesc'] = data3[['Description', 'Asset Name']].agg(' '.join, axis = 1)



def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation]) # Discount punctuation
    tokens = re.split('\W+', txt) # Split words into a list of strings
    txt = [ps.stem(word) for word in tokens if word not in stopwords] #Stem words
    return txt


def tfidf(query):

    tfidf_vect = TfidfVectorizer(analyzer=clean_text)
    corpus = tfidf_vect.fit_transform(data3['A_NameDesc'])
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
    
    
    A_Desc = [data3.loc[i, 'Description'] for i in idx] # Description of CD & KPI
    Name = [data3.loc[i, 'Asset Name'] for i in idx]
    ID = [data3.loc[i, 'Asset ID'] for i in idx]
    
    results = pd.DataFrame({'Asset ID': ID, 'Asset Name':Name, 'Description':A_Desc, 'Match Percentage': matches})

    results = results.to_dict()
    result= {
        "model": "tfidf_asset_reccomendation",
        "results": results
    }
    return result