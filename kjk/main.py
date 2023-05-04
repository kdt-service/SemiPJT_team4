import pandas as pd
from datetime import datetime, timedelta, time
# from scrapy.crawler import CrawlerProcess
# from SearchScraper.SearchScraper.spiders.search_spider import search_spider
from nltk.tokenize import sent_tokenize
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# def run_spider(OUTPUT_FILENAME='day_data.csv'):
#     process = CrawlerProcess()
#     process.crawl(search_spider, output_filename=OUTPUT_FILENAME, output_format='csv')
#     process.start()

def load_data(path):
    data = pd.read_csv(path, encoding='utf-8', dtype=object)
    return data

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

    y = datetime.now().date() - timedelta(days=1)
    y1530 = datetime.combine(y, time(15, 30))
    data = data[data['writed_at'] > y1530]
    return data

def seperate(labeled_data):
    sentences = []
    for idx, row in labeled_data.iterrows():
        article = row['content']
        sentences.append(re.sub('[\,\.]', ' ', row['title']))
        for sentence in sent_tokenize(article):
            sentences.append(re.sub('[\,\.]', ' ', sentence).replace('\ \ ', ' ').strip())

    df = pd.DataFrame({'sentence': sentences})
    df = df[df.sentence.apply(isLonger, args=[5])]
    return df

def pred(text, model, tokenizer):
    input = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=32,
                padding='max_length',
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'# tensor 형식으로 반환
    )
    output = model(
                input['input_ids'], 
                token_type_ids=None, 
                attention_mask=input['attention_mask']
    )
    _, pred = torch.max(output.logits, dim=1)
    return pred

def scoring(df, model_path, model_name):
    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    count = 0
    num = len(df)
    for sen in df['sentence']:
        input = tokenizer.encode_plus(
                    sen,
                    add_special_tokens=True,
                    max_length=32,
                    padding='max_length',
                    pad_to_max_length=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'# tensor 형식으로 반환
        )
        input_ids = input['input_ids'].to(device)
        attention_mask = input['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            pred = outputs.logits.argmax(dim=1).item()
        count += pred
    return count / num

if __name__=='__main__':
    # run_spider('./test.csv', 'csv')

    data = load_data('./SearchScraper/maintest.csv')
    data = preprocess(data)
    for title in data['title']:
        print(title)
    df = seperate(data)
    score = scoring(df, 'final_finbert_model.pth', 'snunlp/KR-FinBert-SC')

    print(score, len(data), len(df))
