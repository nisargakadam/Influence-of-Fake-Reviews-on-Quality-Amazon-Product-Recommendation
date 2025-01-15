import numpy as np
import pandas as pd
import json
import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import json
import nltk
import string
import urllib.request
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# convert jsonl file to text file containing only the reviews
# CHANGED HERE
input_file = "/content/sample_data/Electronics.jsonl"
output_file = "reviews.txt"

num_reviews = 0
max_reviews = 100000

with open(input_file, 'r', encoding='utf-8') as fp, open(output_file, 'w', encoding='utf-8') as out_fp:
	for line in fp:
		if num_reviews >= max_reviews: # Stop reading after 100,000 lines 
			break
		review = json.loads(line.strip())
		if 'text' in review:
			review_text = review['text']
			out_fp.write(review_text + '\n')  # Write the review to the output file
			num_reviews += 1  # Increment review count

with open('reviews.txt', 'r', encoding='utf-8') as text_data:
    tokens = word_tokenize(text_data.read())



# number of sample
print('\n================================\n')
print('Number of sample (number of reviews): ' + str(num_reviews))
print('\n\n')

# number of tokens (not normalized)
print('\n================================\n')
print('Number of tokens (not normalized): ' + str(len(tokens)))
print('\n\n')


# vocabulary size (not normalized)
print('\n================================\n')
print('Vocabulary size (not normalized): ' + str(len(set(tokens))))
print('\n\n')


# number of tokens (normalized)
# convert into lowercase
# remove punctuation and stopwords
tokensLowerCase = []
for word in tokens:
	tokensLowerCase.append(word.lower())

punctuations = string.punctuation

tokensNoPunct= []
for token in tokensLowerCase:
	if token not in punctuations:
		tokensNoPunct.append(token)

stopwordsEnglish = stopwords.words('english')

tokensNormalized = []
for token in tokensNoPunct:
	if token not in stopwordsEnglish:
		tokensNormalized.append(token)


# number of tokens (normalized)
print('\n================================\n')
print('Number of tokens (normalized): ' + str(len(tokensNormalized)))
print('\n\n')

# vocabulary size (normalized)
print('\n================================\n')
print('Vocabulary size (normalized): ' + str(len(set(tokensNormalized))))
print('\n\n')



def remove_non_words(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def convert_jsonl_to_csv(input_file, output_file):
	num_reviews = 0
	max_reviews = 100000
	with open(input_file, 'r', encoding='utf-8') as jsonl_file, open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(['text', 'rating'])

		for line in jsonl_file:
			if num_reviews >= max_reviews:
				break
			review = json.loads(line.strip())
			if 'text' in review and 'rating' in review:
				cleaned_text = remove_non_words(review['text'])
				writer.writerow([cleaned_text, review['rating']])
				num_reviews += 1

	print(f"Data successfully saved to {output_file}")


# Read the text file, assuming it's tab-delimited
df = pd.read_csv("reviews.txt", delimiter="\t")

# Write the dataframe to a CSV file
df.to_csv("reviews.csv", index=False)


input_file = "/content/sample_data/Electronics.jsonl"
output_file = "reviews.csv"
convert_jsonl_to_csv(input_file, output_file)


reviews = pd.read_csv('/content/reviews.csv') #read csv file of reviews in text and ratings (numerical)
#reviews = reviews.dropna(subset=['text']) #dropna values


fake_reviews = pd.read_csv("/content/fake_reviews_dataset.csv")
#fake_reviews.head()
#fake_reviews = fake_reviews.dropna(subset=['text_']) #dropna values





# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Apply preprocessing
fake_reviews['text_'] = fake_reviews['text_'].apply(preprocess_text)
total_tokens = 0
set_total_tokens = 0
for i in range(len(fake_reviews)):
  tokens = word_tokenize(fake_reviews['text_'][i])
  total_tokens += len(tokens)
  set_total_tokens += len(set(tokens))
print(total_tokens)
print(set_total_tokens)


#fake_reviews['text_'][i] = fake_reviews['text_'][i].replace('\n', ' ')



reviews['text'] = reviews['text'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else x)

books_fake_reviews = fake_reviews[fake_reviews["category"] == 'Electronics_5']


books_fake_reviews.loc[:, 'Real'] = np.where(books_fake_reviews['label']== 'CG',0,1)

#books_fake_reviews.tail(5)


vec = CountVectorizer(stop_words='english') #convert words to be features as vectors to be easier to manage

X = vec.fit_transform(books_fake_reviews['text_']) #use vectorized text as X in model

y = books_fake_reviews['Real']  #new label as y


scaler = StandardScaler(with_mean=False)

xscaled = scaler.fit_transform(X)

# get train and test
X_train, X_test, y_train, y_test = train_test_split(xscaled,y, test_size=0.2, random_state=42 )



lf = BaggingClassifier(estimator=SVC(class_weight='balanced', max_iter=300), n_estimators=10, random_state=0)
lf.fit(X_train, y_train)
print('bagging accuracy score: {}'.format(lf.score(X_test, y_test)))


y_pred = lf.predict(X_test)

f1score_lf = f1_score(y_test, y_pred, average = 'weighted')
print(f1score_lf)


model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)


print('decision tree accuracy score: {}'.format(model.score(X_test, y_test)))



y_pred = model.predict(X_test)

f1score_dt = f1_score(y_test, y_pred, average = 'weighted')
print(f1score_dt)



lg_model = LogisticRegression(max_iter=200)
lg_model.fit(X_train, y_train)

print('log regression accuracy score: {}'.format(lg_model.score(X_test, y_test)))


y_pred = lg_model.predict(X_test)

f1score_lg = f1_score(y_test, y_pred, average = 'weighted')
print(f1score_lg)



estimators = [('lg', lg_model), ('svm', lf), ('dt', model)] # 3 model combination
ensemble = VotingClassifier(estimators, voting = 'hard')
ensemble.fit(X_train, y_train)
print('3 model:', ensemble.score(X_test, y_test))


y_pred = ensemble.predict(X_test)

f1score = f1_score(y_test, y_pred, average = 'weighted')

print(f1score)


crtb = classification_report(y_test, y_pred)
print(crtb)
cr = classification_report(y_test, y_pred, output_dict=True)
ensembleCrDf = pd.DataFrame(cr).transpose()
#print(cr)


cr_table = ensembleCrDf.to_markdown(index=False)
print(cr_table)



cr_table = ensembleCrDf.to_latex(index=False, caption="CR", label="tab:classification_report")
print(cr_table)



with open("classification_report.tex", "w") as file:
    file.write(cr_table)




indicies = []
reviews['text'] = reviews['text'].fillna("")
#if review is real then append to new dataset



reviewsVec = vec.transform(reviews['text'])

reviewPred = ensemble.predict(reviewsVec)

reviews['RealPred'] = reviewPred

reviewsVec = vec.transform(reviews['text'])

reviewPred = ensemble.predict(reviewsVec)

reviews['RealPred'] = reviewPred



for i in range(len(reviews)):
  if reviews['RealPred'][i] == 1:
    indicies.append(i)



#print(reviews)
#len(indicies)
#print(indicies)


reviews['RealPred'].value_counts()










