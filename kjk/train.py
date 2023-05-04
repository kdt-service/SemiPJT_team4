import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertConfig, BertForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import json
import time

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
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    train_dataset = MyDataset(train_data, tokenizer)
    test_dataset = MyDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader

def format_time(t):
    h = int(t / 3600)
    m = int((t % 3600) / 60)
    s = int(t % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def accuracy(preds, labels):
    """
    preds: 모델의 예측값
    labels: 정답 레이블
    """
    # 예측값의 최대값이 있는 인덱스를 선택
    preds = torch.argmax(preds, dim=1).flatten()
    # 예측값과 실제값이 동일한 경우 1, 그렇지 않은 경우 0을 반환
    correct = torch.eq(preds, labels).sum()
    # 정확도 계산
    acc = correct.float() / float(len(labels))
    return acc

# 모델 출력으로부터 직접 loss 계산 -> 분류 문제에 더 적합
def train_(train_dataloader, test_dataloader, model_name, num_epochs, learning_rate, start_time):
    # epoch별 평균 train accuracy, loss
    train_acc = []
    train_loss = []
    # 하나의 epoch이 끝날 때마다 test_dataloader로 평가한 결과
    epoch_test = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device :', device)
    # device = torch.device("cpu")

    model = BertForSequenceClassification.from_pretrained(model_name)
    if model_name=='snunlp/KR-FinBert-SC':
        model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=2)
        model.num_labels = 2
    print('pretrained_model :', model_name)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('optimizer : Adam')
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0
        model.train()# model을 학습모드 전환

        
        for i, batch in enumerate(train_dataloader):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, 
                            token_type_ids=None, 
                            attention_mask=attention_mask, 
                            labels=labels)
            
            loss = outputs[0]
            preds = outputs[1]
            acc = accuracy(preds, labels)
            if i % 500 == 0 and i != 0:
                print(f'[{format_time(time.time() - start_time)}] {i}번째 batch - - -')
                print(f'loss : {loss.item()}, acc : {acc.item()}')
            total_loss += loss.item()
            total_acc += acc.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()


        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)
        print(f'[{format_time(time.time() - start_time)}] Epoch {epoch + 1} / {num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')

        train_acc.append(avg_train_acc)
        train_loss.append(avg_train_loss)
        epoch_test.append(evaluate(test_dataloader, model, device))

    print(epoch_test)
    torch.save(model, 'model.pth')
    return train_acc, train_loss, epoch_test

# def flat_accuracy(preds, labels):
#     # aixs=1 줘서 더 높은 라벨 선택
#     # batch별로 묶여있기 때문에 flatten() 써서 1차원으로
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(preds_flat == labels_flat) / len(labels_flat)

# def eval_(test_dataloader):
#     model.eval()
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0

#     for i, batch in enumerate(test_dataloader):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
#         with torch.no_grad():
#             outputs = model(input_ids, 
#                             token_type_ids=None, 
#                             attention_mask=attention_mask)
#         logits = outputs[0]
#         # tensor를 numpy 배열로 변환, 메모리에 남아있지 않도록 detach(), cuda에서 cpu연산으로 변환
#         logits = logits.detach().cpu().numpy()
#         label_ids = labels.to('cpu').numpy()
#         tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#         eval_accuracy += tmp_eval_accuracy
#         nb_eval_steps += 1

#         acc = eval_accuracy/nb_eval_steps
#         if i%100==0 and i!=0:
#             print('Accuracy : {0:.2f}'.format(acc), score/labels.size(0))

#     return accuracy, acc
        
def evaluate(test_dataloader, model, device):
    model.eval()

    total_accuracy = 0
    total_samples = 0

    with torch.no_grad(): # 그래디언트 계산 비활성화
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, 
                            token_type_ids=None, 
                            attention_mask=attention_mask)

            # 모델의 예측 결과 계산
            _, predicted = torch.max(outputs.logits, dim=1)

            # 정확도 계산
            total_accuracy += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_accuracy / total_samples
    print(' '*50 + f'test_data accuracy : {accuracy}')

    model.train()
    return accuracy

def main(model_name, num_epochs=10, learning_rate=2e-5, batch_size=64, num_workers=4):
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained(model_name)

    tr, te = get_dataloader(data, tokenizer, batch_size, num_workers)

    train_acc, train_loss, epoch_test = train_(tr, te, model_name, num_epochs, learning_rate, start_time)

    result_dict = {'model_name': model_name,
                   'optimizer': 'Adam',
                   'batch': batch_size,
                   'learning_late': learning_rate,
                   'train_accuracy': train_acc,
                   'train_loss': train_loss,
                   'epoch': epoch_test,
                   'time_taken': format_time(time.time() - start_time)}
    print(result_dict)

    with open('result.json', 'a') as f:
        f.write(json.dumps(result_dict, indent=4) + '\n')
    print(f'[{format_time(time.time() - start_time)}] 실행 완료' + '\n')

    # 저장된 모델을 불러와서 다시 테스트
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # new_model = torch.load('model.pth')
    # evaluate(te, new_model, device)

    return


if __name__=='__main__':
    data = pd.read_csv('./sen_data.csv')
    # data = data.loc[:10000, :]
    
    main("snunlp/KR-FinBert-SC", num_epochs=15, learning_rate=1e-5)
    # main("snunlp/KR-FinBert-SC")
    # main("bert-base-multilingual-cased")
