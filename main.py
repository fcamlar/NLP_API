from tfidf import tfidf
from flask import jsonify, request
from flask import Flask
import json


app = Flask(__name__)


################## Model Functions ######################

@app.route("/")

@app.route("/api/tfidf",  methods=['GET', 'POST'])
def best_match_asset():
    
    try:
        query = request.json['query']
        
        return tfidf(query)
            
    except:
        return 'Query not submitted'

   



if __name__ == "__main__":
    app.run(debug=True)