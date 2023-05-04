import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, num_labels=2):
        self.data = data
        self.tokenizer = tokenizer
        self.num_labels = num_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']
        
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'# tensor 형식으로 반환
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),# model에 맞는 모양 갖도록
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

def get_dataloader(data, tokenizer, batch_size, num_workers):
    test_dataset = MyDataset(data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_dataloader

def evaluate(test_dataloader, model, device, tokenizer):
    model.eval()

    sen_list = []
    label_list = []
    # correct_by_label = {0: [], 1: []}
    total_accuracy = 0
    total_samples = 0

    for i, batch in enumerate(test_dataloader):
        if i%500 == 0:
            print(i)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad(): # 그래디언트 계산 비활성화
            outputs = model(input_ids, 
                            token_type_ids=None, 
                            attention_mask=attention_mask)

        # 모델의 예측 결과 계산
        _, predicted = torch.max(outputs.logits, dim=1)

        # 예측이 맞은 경우의 데이터 수집
        for i, label in enumerate(labels):
            if predicted[i] == label:
                label_list.append(label.item())
                sen_list.append(tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True))
                # correct_by_label[label.item()].append(tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True))

        # 정확도 계산
        total_accuracy += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_accuracy / total_samples
    print(' '*50 + f'test_data accuracy : {accuracy}')

    df = pd.DataFrame({'sentence':sen_list, 'label':label_list})
    model.train()
    return accuracy, df

def main(data, model_path, model_name):

    model = torch.load(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    te_dl = get_dataloader(data, tokenizer, 64, 4)
    accuracy, df = evaluate(te_dl, model, device, tokenizer)
    print(accuracy)
    df.to_csv(f'./{model_path[:-4]}_result.csv', index=False, encoding='utf-8')

    return

if __name__=='__main__':
    data = pd.read_csv('./sen_data.csv')
    main(data, 'final_finbert_model.pth', 'snunlp/KR-FinBert-SC')
    main(data, 'multilingual_model.pth', 'bert-base-multilingual-cased')
