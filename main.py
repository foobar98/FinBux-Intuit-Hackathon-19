from flask import Flask
from flask import url_for, jsonify, render_template
import company_success as cs

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/foo', methods=['POST'])
def foo():
    # grab reddit data and write to csv
    cs.solve()
    return jsonify({"message": "you're a superstar"})

if __name__ == "__main__":
    app.run(port=8080, debug=True)