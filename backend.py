from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/")
def index():
    return app.send_static_file("index3.html")


@app.route("/talk")
def talk():
    a = request.args.get('userInput', 'no input :O', type=str)
    return jsonify(result=a + " from server")


if __name__ == '__main__':
    app.run(debug=True)