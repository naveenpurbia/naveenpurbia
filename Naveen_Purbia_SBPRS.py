# %%
"""
# Problem Statement
"""

# %%
"""
You are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.
As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

 

In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.

Data sourcing and sentiment analysis \
Building a recommendation system \
Improving the recommendations using the sentiment analysis model \
Deploying the end-to-end project with a user interface 

In this task, you have to analyse product reviews after some text preprocessing steps and build an ML model to get the sentiments corresponding to the users' reviews and ratings for multiple products.
"""

# %%
# Import all the required libraries
import json 
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nlp = en_core_web_sm.load()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import r2_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import time
import os

import warnings
warnings.filterwarnings('ignore')

# %%
# Read the data file and get column details
product_df = pd.read_csv("sample30.csv", sep=",")
product_df.info()


# %%
#Check the data in csv
product_df.head(5)

# %%
# Check counts for the 5 ratings.
product_df['reviews_rating'].astype('category').value_counts()

# %%
#Review by user city
product_df['reviews_userCity'].value_counts()

# %%
# Get overview of rating and sentiment
product_df.groupby(['reviews_rating','user_sentiment']).size()

# %%
# Get total of Positive and Negative count
product_df['user_sentiment'].astype('category').value_counts()

# %%
# Check columns for % of missing values
product_df.isna().sum() / len(product_df) * 100

# %%
# Drop reviews_userCity and reviews_userProvince as these have huge missing values
product_df = product_df.drop(['reviews_userCity','reviews_userProvince'], axis=1)

# %%
"""
## EDA
"""

# %%
# Derive date from timestamp and derive month and year
product_df['date'] = product_df['reviews_date'].str[:10]
product_df['date'] = pd.to_datetime(product_df['date'], errors='coerce').dt.date
product_df['date'].fillna(product_df['date'].value_counts().idxmax(),inplace=True)
product_df['date'] = pd.to_datetime(product_df.date, format='%Y-%m-%d')
product_df['month'] = product_df['date'].dt.month
product_df['year'] = product_df['date'].dt.year
product_df['date'].head()

# %%
#product_df['Year'].value_counts()
product_df.head(1)

# %%
# Plot a graph of Month vs Rating for better visualization
month_by_review =product_df.groupby(['month'])['reviews_rating'].count()
month_by_review = pd.DataFrame(month_by_review)
month_by_review.plot.bar()

# %%
# Convert to dataframe
month_by_review = pd.DataFrame(month_by_review)

# %%
# Plot a graph of Month vs Rating for better visualization
month_by_review.plot.bar()

# %%
# Plot a graph of Year vs Rating for better visualization
year_by_review =product_df.groupby(['year'])['reviews_rating'].count()
year_by_review = pd.DataFrame(year_by_review)
year_by_review.plot.bar()

# %%
# Plot a graph of products with Positive reviews where users have bought those products
top_positive_review_products=product_df[(product_df['user_sentiment']=='Positive') & ( product_df['reviews_didPurchase'] == True )]
top_positive_review_products=top_positive_review_products.groupby(['name'])['reviews_rating'].count().reset_index()
top_positive_review_products.sort_values(by='reviews_rating',ascending=False).head(10).plot.bar(x='name',y='reviews_rating')


# %%
# Plot a graph of products with Positive reviews where users have bought not necessarily bought those products
top_positive_review_products=product_df[product_df['user_sentiment']=='Positive']
top_positive_review_products=top_positive_review_products.groupby(['name'])['reviews_rating'].count().reset_index()
top_positive_review_products.sort_values(by='reviews_rating',ascending=False).head(10).plot.bar(x='name',y='reviews_rating')


# %%
# Plot a graph of products with Negative reviews where users have bought those products
top_negative_review_products=product_df[(product_df['user_sentiment']=='Negative') & ( product_df['reviews_didPurchase'] == True )]
top_negative_review_products=top_negative_review_products.groupby(['name'])['reviews_rating'].count().reset_index()
top_negative_review_products.sort_values(by='reviews_rating',ascending=False).head(10).plot.bar(x='name',y='reviews_rating')


# %%
# Plot a graph of products with Negative reviews where users have bought not necessarily bought those products
top_negative_review_products=product_df[product_df['user_sentiment']=='Negative']
top_negative_review_products=top_negative_review_products.groupby(['name'])['reviews_rating'].count().reset_index()
top_negative_review_products.sort_values(by='reviews_rating',ascending=False).head(10).plot.bar(x='name',y='reviews_rating')


# %%
# Plot of users who have given rating and purchased it
product_df['reviews_didPurchase'].value_counts().plot.bar()

# %%
# As the total do not come around 30,000 there looks to be lot of blank values.
# Replace it will null
product_df['reviews_didPurchase'].fillna('Null',inplace=True)

# %%
product_df['reviews_didPurchase'].value_counts().plot.bar()

# %%
"""
## There are lot of users who have not purchased products but given review.
### Also many users have not given reviews if they have purchased it. Hence we can drop this column reviews_didPurchase after completing EDA. Ihas almost ~50% missing values.
"""

# %%
# USers who have purchased and recommended products
product_df['reviews_doRecommend'].value_counts().plot.bar()

# %%
# As the total do not come around 30,000 there looks to be some blank values.
# Replace it will null
product_df['reviews_doRecommend'].fillna('Null',inplace=True)

# %%
product_df['reviews_doRecommend'].value_counts().plot.bar()

# %%
# Compare recommendation vs rating
product_df.groupby(['reviews_doRecommend','reviews_rating'])['reviews_doRecommend'].count()

# %%
"""
### We see ratings are mostly >3 where user have recommended products and mostly <=3 where they have not recommended
"""

# %%
# Check user_recommendation vs user_sentiment
product_df.groupby(['reviews_doRecommend','user_sentiment'])['reviews_doRecommend'].count()

# %%
"""
### Product sentiment is usually more Positive when users have recommended it.
"""

# %%
# Check user_rating vs user_sentiment
product_df.groupby(['reviews_rating','user_sentiment'])['reviews_rating'].count()

# %%
"""
### For each rating, there are both positive and negative reviews. But higher ratings have more % of Positive sentiment than lower ratings.
 Here many products have Positive sentiment but rating of 1 and 2. Similarly many products have Negative sentiment, but rating of 3,4,5.
 
"""

# %%
#Drop reviews_didPurchase which have many null values (~50%)
product_df = product_df.drop(['reviews_didPurchase'], axis=1)
product_df.head(2)

# %%
# Get length of review text for each review
product_df['review_length']=product_df['reviews_text'].apply(lambda x : len(x.split()))

# %%
# Plot relation between Rating Vs Review LEngth
sns.scatterplot(x='reviews_rating', y='review_length', data=product_df)

# %%
"""
### Higher ratings have usually lengthy review than lower rating
"""

# %%
# compare the review text length
product_df.groupby(pd.cut(product_df.review_length, np.arange(0,330,30))).count()

# %%
# Create the categories based on review length
def review_range(row):
    if row['review_length'] <= 50:
        val = '0-50'
    elif (row['review_length'] > 50 and row['review_length'] <= 150):
        val = '50-150'
    elif (row['review_length'] > 150 and row['review_length'] <= 300):
        val = '150-300'
    elif row['review_length'] > 300:
        val = '> 300'
    return val

# %%
product_df['review_category'] = product_df.apply(review_range, axis=1)

# %%
product_df.head(3)

# %%
plt.figure (figsize= (8,10))
sns.barplot(x = "reviews_rating", y = "review_length", hue = "review_category", data = product_df)
plt.show()

# %%
"""
### There are more users who have given long reviews
"""

# %%
"""
# Data Preprocessing
"""

# %%
# Find out which user have given most reviews
product_df['reviews_username'].value_counts().head(5)

# %%
top_user_review = product_df[product_df['reviews_username']=='byamazon customer']

# %%
# Same user has given different ratings for same product.
top_user_review.groupby(['name','reviews_rating'])['reviews_rating'].count()

# %%
#product_df['average_rating']=product_df.groupby(['id','reviews_username'])['reviews_rating'].transform('mean').round(1)

# %%
# Write function to clean the text, convert to lowercase and remove all the unnecessary elements.
def clean_text(text):
    
    # make the text lowercase
    text = text.lower()
    lstcleaned = []
    for word in text.split():
        
        # remove text in square bracket
        word = re.sub('\[.*?\]', '', word)
        
        # remove punctuation
        word = re.sub("[^\w\s_']", '', word)
        
        # remove numbers
        if not(re.search("\d+", word) and re.search("[a-zA-Z]", word)):
            lstcleaned.append(word)

    text = " ".join(lstcleaned)
    return text


# %%
#Lemmatize the texts
def lemma_text(text):    
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# %%
# Keep only POS tags noun,pronoun,verb and adjective in the reviews.
def getNNtags(text):
    doc1 = nlp(text)
    lstNNtags = []
    for tok in doc1:
        if tok.pos_ == "NOUN" or tok.pos_ == 'PRON' or tok.pos_ == 'ADJ'  or tok.pos_ == 'ADV':
            lstNNtags.append(tok.text)
    return " ".join(lstNNtags)


# %%
df_clean = product_df.copy()

# %%
# Apply all the above data preprocessing tasks.
df_clean['clean_text'] = df_clean['reviews_text'].apply(clean_text)
df_clean['lemma_review'] = df_clean['reviews_text'].apply(lemma_text)
df_clean['pos_removed'] = df_clean['reviews_text'].apply(getNNtags)
df_clean.head()

# %%
 # Remove all the stopwords in English language
stop = stopwords.words('english')
df_clean['stop_removed'] = df_clean['pos_removed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# %%
# View the cleaned reviews by checking full text
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
df_clean.head(3)

# %%
# Check top 30 words in reviews using wordcloud
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import WordCloud
stoplist = set(stopwords.words("english"))

wordcloud = WordCloud(stopwords=stoplist,max_words=30).generate(str(df_clean['stop_removed']))

print(wordcloud)
#fig = plt.figure(1)
plt.figure (figsize= (11,11))
plt.imshow(wordcloud)
plt.axis('off')
plt.show();

# %%
"""
## Feature Extraction
"""

# %%
#Keeping only the relevant columns which are required for product recommendation
df_feature=df_clean[['reviews_username','name','stop_removed','reviews_rating','user_sentiment']]
df_feature.info()

# %%
# Convert user sentiment as 1 for Positive and 0 for NEgative
df_feature['stop_removed'] = df_feature['stop_removed'].astype(str)
df_feature['user_sentiment']=df_feature['user_sentiment'].map(lambda x : 1 if x=='Positive' else 0)

# %%
df_feature.info()

# %%
#create pickle file
import pickle as pickle
pickle.dump(df_feature, open("data.pkl","wb"))

# %%
df_feature.head(1)

# %%
# Feature extraction using tf-idf
tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,2))#,analyzer='word',stop_words= 'english')
tf_idf_vectorizer.fit(df_feature['stop_removed'])
X = tf_idf_vectorizer.transform(df_feature['stop_removed'])
y = df_feature['user_sentiment']

# %%
#Saving tf-idf vectorizer in pickel file format
pickle.dump(tf_idf_vectorizer,open('tf_idf_vectorizer.pkl','wb'))
#loading pickle object
tf_idf_vectorizer = pickle.load(open('tf_idf_vectorizer.pkl','rb'))

# %%
#Lets split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)


# %%
#Checking Class Imbalance 

df_feature.groupby(['user_sentiment']).count()

# %%
X_train

# %%
# Since there are more Positive reviews than Negative reviews, we need to handle class imbalance.
from collections import Counter
from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import RandomizedSearchCV
counter = Counter(y_train)
print("Before", counter)


#oversampling using SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

counter = Counter(y_train_smote)
print("After", counter)

# %%
"""
#### Now the class imbalance is corrected.
"""

# %%
# Create function to calculate metrics for test and train data.
def mymetric(y_test, y_test_pred):
    confusion = metrics.confusion_matrix( y_test, y_test_pred )
    print("\n\nConfusion: \n", confusion)

    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print("\n\nOverall accuracy: ", accuracy)

    f1score = f1_score(y_test, y_test_pred, average='weighted')
    print("\n\nf1_score: ", f1score)
    
    return [accuracy, f1score]

# %%
"""
#### Let's start creating all the models.
"""

# %%
"""
### Logistic REgression
"""

# %%
# Logistic Regression Model
lr = LogisticRegression(multi_class='multinomial', solver='saga')
lr.fit(X_train, y_train)

# %%
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# %%
df_metric = pd.concat([df_metric, \
                      pd.DataFrame([mymetric(y_train, y_train_pred)], \
                                   columns=['Accuracy', 'F1Score'], index=['Logistic_Regression_Train'])])

# %%
df_metric = pd.concat([df_metric, \
                      pd.DataFrame([mymetric(y_test, y_test_pred)], \
                                   columns=['Accuracy', 'F1Score'], index=['Logistic_Regression_Test'])])

# %%
df_metric

# %%
# Since random forest model gives best performance out of 3, we will select this.
pickle.dump(lr, open('logistic_regression_model.pkl', 'wb'))
final_ml_model =  pickle.load(open('logistic_regression_model.pkl', 'rb'))

# %%
"""
## Recommendation System
"""

# %%
# Get the original data
product_df=pd.read_csv("sample30.csv", sep=',')

# %%
# Get the important columns
recom_df=product_df[['name','reviews_username','reviews_rating']]

# %%
recom_df.info()

# %%
# Remove rows with no user listed
recom_df = recom_df[~recom_df.reviews_username.isna()]

# %%
recom_df.info()

# %%
#Split the data into train and test datasets

train, test = train_test_split(recom_df, test_size=0.30, random_state=12)

print(train.shape)
print(test.shape)

# %%
# Create a user vs product matrix
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating',
).fillna(0)


# %%
df_pivot.head(4)

# %%
dummy_train = train.copy()
dummy_train.head(5)

# %%
dummy_train['reviews_rating'].value_counts()

# %%
#The products not rated by user is marked as 1 for prediction. 

dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# %%
# Create matrix and fill all missing values with 1
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)


dummy_train.head()

# %%
"""
## User based similarity
"""

# %%
#User Similarity Matrix via pairwise_distance function

user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

# %%
"""
## Adjusted Cosine
"""

# %%
# Create a user-product matrix.

df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)

# %%
df_pivot

# %%
# Take inverse of matrix and subtract from mean.
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

df_subtracted.head()

# %%
#Creating the User Similarity Matrix using pairwise_distance function

user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)

# %%
user_correlation.shape

# %%
"""
### Prediction - User User
"""

# %%
# Replace negative average with zeros as we need to predict positively related users.
user_correlation[user_correlation<0]=0
user_correlation

# %%
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings

# %%
user_predicted_ratings.shape

# %%
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()

# %%
user_final_rating.head(20)

# %%
#Lets take a random user ID from the given dataset as input
user_input='joshua'
print(user_input)
#Top 20 recommendations
df_user_user = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
df_user_user

# %%
# saving the model
pickle.dump(user_final_rating.astype('float32'), open('user_final_rating.pkl', 'wb'))
final_user_recommendation =  pickle.load(open('user_final_rating.pkl', 'rb'))

user_input='joshua'
print(user_input)

# %%
df_user_match = final_user_recommendation.loc[user_input].sort_values(ascending=False)[0:20]

# %%
type(df_user_match)

# %%
# Final output after getting top 5 products whose average rating is highest from 20 recommended products.
def id_to_mean(product_name):
    df_id = df_feature.loc[df_feature['name']==product_name]
    X_id = tf_idf_vectorizer.transform(df_id['stop_removed'])
    output = final_ml_model.predict(X_id)
    return np.mean(output)
# product and product sentiment
top_5_products= dict()
for prod in df_user_match.index.tolist():
    top_5_products[prod]=id_to_mean(prod)

# %%
# Get top 5 products from above matrix
top_5_products = dict(sorted(top_5_products.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:5])

# %%
# Name of top 5 products.
top_5_products

# %%
