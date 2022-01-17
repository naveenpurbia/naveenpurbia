# import Flask class from the flask module
from flask import Flask, request, jsonify, render_template
import pandas as pd

import numpy as np
import pickle

user_final_rating = pd.read_pickle("user_final_rating.pkl")
df_feature = pd.read_csv("sample30.csv")
tf_idf_vectorizer = pickle.load(open('tf_idf_vectorizer.pkl', 'rb'))
final_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

def recommend(user_input):
    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
    # Based on positive sentiment percentage.
    i= 0
    a = {}
    for prod_name in d.index.tolist():
        df_id = df_feature.loc[df_feature['name']==prod_name]
        X_id = tf_idf_vectorizer.transform(df_id['stop_removed'])
        final_model.predict(X_id)
        a[prod_name] = final_model.predict(X_id).mean()*100
    b= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    return b


# Create Flask object to run
app = Flask(__name__,template_folder='templates')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    username = str(request.form.get('reviews_username'))
    print(username)
    prediction = recommend(username)
    print("Output :", prediction)
    return render_template('index.html', prediction_text='Top 5 Recommendations are:\n {}'.format(prediction))
    #return prediction[0]


if __name__ == "__main__":
    app.run(debug = True)
