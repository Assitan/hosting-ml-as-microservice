from flask import Flask, render_template, request
from predict_sentiment_analysis import get_sentiment
 
app = Flask(__name__)

@app.route('/')
def hello_whale():
    return 'Whale, Hello there!'
 
 
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        input = request.args.get('input')
    else:
        input = request.get_json(force=True)['input']
    if not input:
        return 'No input value found'
    return get_sentiment(input)
 
if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0')
    
