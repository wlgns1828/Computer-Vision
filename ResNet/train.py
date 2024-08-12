import torch
import torch.nn as nn
import torch.optim as optim
from model import *
import time
import copy
import os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 학습 데이터를 csv파일로 저장
def save_history_to_csv(history, filename):
    df = pd.DataFrame(history, columns=['value'])
    df.to_csv(filename, index=False)
# learning rate 출력
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
# corrects 개수 출력    
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects
# loss 값 출력
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)
    # 평가 중에는 업데이트 하지 않음
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b
# epoch마다 loss와 accuracy를 출력
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric
# 학습
def train_val(model, params):
    # 파라미터 설정
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    # 손실과 정확도를 저장할 리스트
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}
    train_time = {'train_time': []}
    # 모델 가중치를 저장
    best_model_wts = copy.deepcopy(model.state_dict())
    # 첫 손실 값은 무한대로 설정
    best_loss = float('inf')
    # 학습 시작한 시간 저장
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch+1, num_epochs, current_lr))
        # 학습
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)
        # 평가
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)
        # 가장 좋은 손실 값을 저장
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            # best_loss일 때 가중치를 저장
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')
            print('Get best val_loss')

        lr_scheduler.step()

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)
        train_t = (time.time()-start_time)/60
        train_time['train_time'].append(train_t)
    model.load_state_dict(best_model_wts)
    
    return model, loss_history, metric_history, train_time

def save_all(loss_history, metric_history, save_dir, train_time):
    save_history_to_csv(loss_history['train'], os.path.join(save_dir, 'train_loss.csv'))
    save_history_to_csv(loss_history['val'], os.path.join(save_dir, 'val_loss.csv'))
    save_history_to_csv(metric_history['train'], os.path.join(save_dir, 'train_metric.csv'))
    save_history_to_csv(metric_history['val'], os.path.join(save_dir, 'val_metric.csv'))
    save_history_to_csv(train_time['train_time'], os.path.join(save_dir, 'train_time.csv'))
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')


