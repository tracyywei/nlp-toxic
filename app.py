from flask import Flask, render_template, request
import numpy as np
import re
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import re
import string

#Null cleaning function

def myfillna(series):
    if series.dtype is pd.np.dtype(float):
        return series.fillna('')
    elif series.dtype is pd.np.dtype(int):
        return series.fillna('')
    else:
        return series.fillna('NA')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

data=pd.read_csv('train.csv',encoding='latin-1')
data=data.apply(myfillna)

RE_PREPROCESS = r'\W+|\d+' #the regular expressions that matches all non-characters
data.text = np.array( [ re.sub(RE_PREPROCESS, ' ', comment).lower() for comment in data.text])

train_size = int(len(data) * .2)
train_posts = data['text'][train_size:]
train_tags = data['toxic'][train_size:]

vect=TfidfVectorizer(tokenizer=LemmaTokenizer(),stop_words='english',ngram_range=(1, 2),max_df=0.7,min_df=30)
vect.fit(train_posts)

x_train=vect.transform(train_posts)

logreg = LogisticRegression(penalty='l1', solver='liblinear')
logreg.fit(x_train, train_tags)

app = Flask(__name__)
@app.route('/', methods=['POST','GET'])
def main():
	if request.method == 'GET':
		return render_template('index.html')

	if request.method == 'POST':
		review = request.form['review']
		corpus = []
		review = re.sub('[^a-zA-Z]', ' ', review)
		review = review.lower()
		review = review.split()
		lemmatizer = WordNetLemmatizer()
		review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
		review = ' '.join(review)
		corpus.append(review)
		x_tfid = vect.transform(corpus).toarray()
		answer = logreg.predict(x_tfid)
		answer = str(answer[0])
		if answer == '1':
			return "The comment is negative. You cannot post this"
		else:
			return "Thanks for the comment"


if __name__ == "__main__":
	app.run(host='0.0.0.0',port='8001',debug=True)
