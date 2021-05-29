# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'mnb_model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv_for_transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     if request.method == 'POST':
# 		message = request.form['message']
# 		data = [message]
#     	vect = cv.transform(data).toarray() ##converting data to vectors for prediction
#     	my_prediction = classifier.predict(vect)
#
# 		if my_prediction == 1:
# 			return render_template('index.html', prediction_text="Gotch!! This is a Spam Message.")
# 		else:
# 			return render_template('index.html', prediction_text="This is a Ham(Normal) Message.")
# 	else:
#          return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray() ##converting data to vectors for prediction
		my_prediction = classifier.predict(vect)

		if my_prediction == 1:
			return render_template('index.html', prediction_text="Gotch!! This is a Spam Message.")
		else:
			return render_template('index.html',prediction_text="This is a Ham(Normal) Message.")

	else:
		return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True)