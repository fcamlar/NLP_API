from tfidf import tfidf
from links import links
from esrc_bert import bert
from tfidf_bert import bert_answers
from flask import jsonify, request
from flask import Flask
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

################## Model Functions ######################

@app.route("/")

@app.route("/api/tfidf",  methods=['GET', 'POST'])
def best_match_asset():
    
    try:
        query = request.json['query']
        
        return tfidf(query)
            
    except:
        return 'Query not submitted'

@app.route("/api/links",  methods=['GET', 'POST'])
def best_match_links():
    
    try:
        query = request.json['query']
        
        return links(query)
            
    except:
        return 'Query not submitted'

   
@app.route("/api/bert", methods=['POST'])
def esrc_bert():
    
    try:
        question = request.json['query']

        return bert(question)
    except:
        return 'Question not submitted'

@app.route("/api/bert_long", methods=['POST'])
def esrc_bert_long():
    
    try:
        question = request.json['query']
        no_answers = request.json['no_answers']
        
        return bert_answers(question, no_answers)
    except:
        return 'Question not submitted'


if __name__ == "__main__":
    app.run(debug=True)