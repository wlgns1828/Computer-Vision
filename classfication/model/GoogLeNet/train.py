import os
import torch
from load_dataset import training_data_loader, test_data_loader
from model import GoogLeNet
import torch.optim as optim
from set_parameter import get_parser
import time
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from graph import graph

parser = get_parser()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        output, aux1, aux2 = outputs

        output_loss = criterion(output, labels)
        aux1_loss = criterion(aux1, labels)
        aux2_loss = criterion(aux2, labels)
        loss = output_loss + 0.3*(aux1_loss + aux2_loss)
            
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

if __name__ == '__main__': 
    model = GoogLeNet(aux_logits=True, num_classes=parser.n_class, init_weights=True).to(device)    
    print('GoogLeNet created')
    
    optimizer = optim.Adam(params=model.parameters(), lr=parser.lr)
    print('Optimizer created')
    print("="*50)
    print("\n")

    criterion = nn.CrossEntropyLoss()

    # 손실과 정확도를 저장할 리스트
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}
    train_time = {'train_time': []}
    start_time = time.time()

    num_epochs = parser.n_epochs
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):
    
        
        train_loss, train_accuracy = train(model, optimizer, criterion, training_data_loader, device)
        val_loss, val_accuracy = evaluate(model, criterion, test_data_loader, device)

        # 손실과 정확도 저장
        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)
        metric_history['train'].append(train_accuracy)
        metric_history['val'].append(val_accuracy)

        # 에포크 동안의 시간을 측정
        epoch_time = (time.time() - start_time)/60
        train_time['train_time'].append(epoch_time)
        print(f'Epoch [{epoch+1}/{num_epochs}]')    
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        print("\n")



# 학습 데이터를 csv파일로 저장
def save_history_to_csv(history, filename):
    df = pd.DataFrame(history, columns=['value'])
    df.to_csv(filename, index=False)
    
def save_all(loss_history, metric_history, save_dir, train_time):
    save_history_to_csv(loss_history['train'], os.path.join(save_dir, 'train_loss.csv'))
    save_history_to_csv(loss_history['val'], os.path.join(save_dir, 'val_loss.csv'))
    save_history_to_csv(metric_history['train'], os.path.join(save_dir, 'train_metric.csv'))
    save_history_to_csv(metric_history['val'], os.path.join(save_dir, 'val_metric.csv'))
    save_history_to_csv(train_time['train_time'], os.path.join(save_dir, 'train_time.csv'))

os.makedirs(parser.result_path, exist_ok=True)
save_dir = parser.result_path   
save_all(loss_history, metric_history, save_dir, train_time)
graph()