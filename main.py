from flask import Flask
from flask import url_for, jsonify, render_template, request
import train as train_script
import test as test_script

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bar', methods=['POST'])
def bar():
    train_script.model_train()
    f = open("first_run.txt","r")
    if f.mode == 'r':
        contents =f.read()

    return contents

@app.route('/but1', methods=['POST'])
def but1():
    newVal = int(request.form['newVal'])
    test_script.test_data(newVal,26)
    f = open("model_output.txt","r")
    if f.mode == 'r':
        contents =f.read()

    return contents

@app.route('/but2', methods=['POST'])
def but2():
    newVal = int(request.form['newVal'])
    test_script.test_data(newVal,33)
    f = open("model_output.txt","r")
    if f.mode == 'r':
        contents =f.read()

    return contents

@app.route('/but3', methods=['POST'])
def but3():
    newVal = int(request.form['newVal'])
    test_script.test_data(newVal,45)
    f = open("model_output.txt","r")
    if f.mode == 'r':
        contents =f.read()

    return contents

@app.route('/but4', methods=['POST'])
def but4():
    newVal = int(request.form['newVal'])
    test_script.test_data(newVal,4)
    f = open("model_output.txt","r")
    if f.mode == 'r':
        contents =f.read()

    return contents

@app.route('/but5', methods=['POST'])
def but5():
    newVal = int(request.form['newVal'])
    test_script.test_data(newVal,10)
    f = open("model_output.txt","r")
    if f.mode == 'r':
        contents =f.read()

    return contents


if __name__ == "__main__":
    app.run(port=8081, debug=True)