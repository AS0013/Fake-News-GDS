import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'

data = pd.read_csv(url)




def cleanText(text):
    # lower case
    text = text.lower()

    # should not contain multiple spaces, tabs or newlines
    text = re.sub(r'\s+', ' ', text)

    # date and time stuff

    # replace numbers with <NUM>
    # text = re.sub(r'\d+', '<NUM>', text)

    # replace urls with <URL>
    text = re.sub(r'(http|https)://[^\s]*', '<URL>', text)

    # replace emails with <EMAIL>
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', '<EMAIL>', text)

    return text

data['content'] = data['content'].apply(cleanText)

data.to_csv('cleaned_data.csv', index=False)




