import pandas as pd
from nltk.tokenize import sent_tokenize
import re
from datetime import datetime

def isLonger(text, length):
    if len(text) > length:
        return True
    else:
        return False

def preprocess(data):
    data.dropna(subset=['content', 'title'], inplace=True)
    data.drop_duplicates(subset=['title', 'writer'], keep='first', inplace=True)
    data.drop_duplicates(subset=['content'], keep='first', inplace=True)

    data = data[data.content.apply(isLonger, args=[100])]
    data = data[data.title.str.contains('SK') | data.title.str.contains('하이닉스')]

    data['content'] = data['content'].str.replace('↑', '상승', regex=True).replace('↓', '하락', regex=True)
    data['content'] = data['content'].str.replace('[^가-힣a-zA-Z0-9 \.\,]', ' ', regex=True).replace(' +', ' ', regex=True)
    data['title'] = data['title'].str.replace('↑', '상승', regex=True).replace('↓', '하락', regex=True)
    data['title'] = data['title'].str.replace('[^가-힣a-zA-Z0-9 ]', ' ', regex=True).replace(' +', ' ', regex=True)

    data['writed_at'] = pd.to_datetime(data['writed_at'], format='%Y-%m-%d %H:%M:%S')
    return data

def labeling(data, stock_data):
    stock_data['date'] = pd.to_datetime(stock_data['날짜'])
    stock_data['up'] = None

    # 전일대비 상승인지 하락인지 1, 0 부여
    for i in range(len(stock_data)-1):
        stock_data.loc[i, 'up'] = int(stock_data.loc[i, '종가'] > stock_data.loc[i+1, '종가'])

    data['label'] = None
    for i in range(len(stock_data)-1):
        end = stock_data.loc[i, 'date'] + pd.Timedelta(hours=15, minutes=30)
        start = stock_data.loc[i+1, 'date'] + pd.Timedelta(hours=15, minutes=30)
        data.loc[(start < data['writed_at'])&(data['writed_at'] < end), 'label'] = stock_data.loc[i, 'up']

    labeled_data = data.loc[:, ['content', 'title', 'writed_at', 'label', 'url']]
    labeled_data.dropna(inplace=True)
    # labeled_data.to_csv('./labeled_data.csv', index=False)
    return labeled_data

def seperate(labeled_data):
    sentences = []
    labels = []
    for idx, row in labeled_data.iterrows():
        article = row['content']
        article_label = row['label']
        sentences.append(re.sub('[\,\.]', ' ', row['title']))
        labels.append(article_label)
        for sentence in sent_tokenize(article):
            sentences.append(re.sub('[\,\.]', ' ', sentence).replace('\ \ ', ' ').strip())
            labels.append(article_label)

    df = pd.DataFrame({'sentence': sentences, 'label': labels})
    df = df[df.sentence.apply(isLonger, args=[5])]
    return df

if __name__=='__main__':
    data = pd.read_csv('C:/Users/tlrks/Desktop/workspace/project/SemiProject/SearchScraper/test_data.csv')
    stock_data = pd.read_csv('./test_stock_data.csv')
    preprocessed_data = preprocess(data)
    labeled_data = labeling(preprocessed_data, stock_data)
    df = seperate(labeled_data)
    df.to_csv('./test_sen_data.csv', index=False)