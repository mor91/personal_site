from flask import Flask, jsonify, request
import spacy
import tensorflow as tf
from chatbot import create_model, decode

app = Flask(__name__)
nlp = spacy.load('en', parser=False, tagger=False, entity=False)

csess = tf.Session()
imodel = create_model(csess, True)
imodel.batch_size = 1  # We decode one sentence at a time.


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/talk")
def talk():
    a = request.args.get('userInput', 'no input :O', type=str)
    return jsonify(result=decode(a, csess, imodel, nlp))


if __name__ == '__main__':
    app.run()
