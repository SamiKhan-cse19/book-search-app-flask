from flask import Flask, jsonify, request
from utils import get_results

app = Flask(__name__)

@app.post('/bookSearch')
def movie_search():
    data = request.json

    # error checking for input text
    try:
        sample = data['query']
    except KeyError:
        return jsonify({'error': 'No query sent'})
    
    queries = [sample]
    predictions = get_results(queries)

    # error checking for server response
    try:
        result = jsonify(predictions)
    except TypeError as e:
        result = jsonify({'error': str(e)})
    return result

if __name__ == '__main__':
    # start development server
    app.run(host='0.0.0.0', debug=True)