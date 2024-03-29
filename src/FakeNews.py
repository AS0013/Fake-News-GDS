import re
import pandas as pd
import nltk
# nltk.download('punkt') ----> udkommenter denne linje, hvis du ikke har nltk installeret
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import PorterStemmer


url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'

data = pd.read_csv(url)

data.to_csv('originaldata.csv', index=False)

print("the shape of the data is: ", data.shape)

print(data.head())




def cleanText(text):
    # lower case
    text = text.lower()

    # should not contain multiple spaces, tabs or newlines
    text = re.sub(r'\s+', ' ', text)

    # date and time stuff
    # text = re.sub(r'\b(?:the )?(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\b', '<DATE>', text)
    # text = re.sub(r'r"([a-zA-Z]{3}\s\d{1,2}\s\d{4})"', '<DATE>', text, flags=re.IGNORECASE)
    # text = re.sub(r'\b(?:the )?(\d{1,2})(?:st|nd|rd|th)?(?: of)?(?: (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))(?: (\d{4}|\d{2}))?\b', '<DATE>', text)
    # text = re.sub(r'\b(?:the )?(\d{1,2})(?:st|nd|rd|th)?(?: of)?(?: (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))(?: (\d{4}|\d{2}))?\b', '<DATE>', text)

    # replace dates with <DATE>
    #  january 18, 2018. jan 18, 2018. 2018-01-18
    date_pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s+|\s+)\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'

    # date_pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
    text = re.sub(date_pattern, '<DATE>', text)
    # nov. 5
    date_pattern2 = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.\s+\d{1,2}\b'
    text = re.sub(date_pattern2, '<DATE>', text)

    # text = re.sub(r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s*(?:\d{1,2}[-/th|st|nd|rd\s]*))?(?:\s*(?:\d{4}|\d{2}))?\b', '<DATE>', text)


    # replace numbers with <NUM>
    text = re.sub(r'\d+', '<NUM>', text)

    # replace urls with <URL>
    text = re.sub(r'(http|https)://[^\s]*', '<URL>', text)

    # replace emails with <EMAIL>
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', '<EMAIL>', text)

    # remove punctuation and commas

    text = re.sub(r'[.,;!?]', '', text)

    return text

data['content'] = data['content'].apply(cleanText)

data.to_csv('cleaned_data.csv', index=False)

# tokenize the text using nltk

tokens = data['content'].apply(nltk.word_tokenize)

print(tokens.head())

# remove stop words

stop_words = set(stopwords.words('english'))

filtered_tokens = []

for token in tokens:
    filtered_words = []
    for word in token:
        if word not in stop_words:
            filtered_words.append(word)
    filtered_tokens.append(filtered_words)

print(filtered_tokens)
print("len of filtered tokens: ", len(filtered_tokens))

# print(tokens.shape)
