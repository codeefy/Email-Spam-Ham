from flask import Flask, render_template, request ,jsonify
import pickle

app = Flask(__name__)
cv = pickle.load(open("D:\Email-Spam-Ham\Email-Spam-Project-main\cv.pkl", 'rb'))
clf = pickle.load(open("D:\Email-Spam-Ham\Email-Spam-Project-main\clf.pkl", 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    email = request.form.get('email')
    # predict email
    print(email)
    X = cv.transform([email])
    prediction = clf.predict(X)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html', response=prediction)

@app.route("/api/predict",methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    email = data['content']
    X = cv.transform([email])
    prediction = clf.predict(X)
    prediction = 1 if prediction == 1 else -1
    return jsonify({'prediction': prediction , 'email': email})

if __name__ == "__main__":
    app.run(debug=True)
