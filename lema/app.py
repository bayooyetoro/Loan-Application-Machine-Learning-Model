from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv(request.files.get('file'))

    predictions = model.predict(data)
    approval_rate = (predictions == 1).mean()
    refusal_rate = (predictions == 0).mean()
    if approval_rate == 1:
        approval_rate = "Yes"
        refusal_rate="No"
    if refusal_rate == 1:
        refusal_rate = "Yes"
        approval_rate ="No"

    return render_template('results.html', approval_rate=approval_rate, refusal_rate=refusal_rate)


if __name__ == '__main__':
    app.run(debug=True)