from model import *
from train import createFolder, save_all, train_val
from load_dataset import train_dl, val_dl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

# gpu 및 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18().to(device)
#model = models.resnet18(pretrained = False).to(device)


loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
#opt = optim.Adam(model.parameters(), lr=0.001)

#lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)
lr_scheduler = StepLR(opt, step_size=10, gamma=0.1)

# 손실 값과 정확도를 저장할 경로 설정
save_dir = './training_history_18(cifar10)'

# 폴더 생성
createFolder(save_dir)
createFolder('./models')

# 파라미터 설정
def get_params_train():
    params_train = {
        'num_epochs':50,
        'optimizer':opt,
        'loss_func':loss_func,
        'train_dl':train_dl,
        'val_dl':val_dl,
        'sanity_check':False,
        'lr_scheduler':lr_scheduler,
        'path2weights':'./models/weights_34(cifar10).pt',
    }
    return params_train


params_train = get_params_train()

# 학습
model, loss_hist, metric_hist, train_time = train_val(model, params_train)

# 결과 저장
save_all(loss_hist, metric_hist, save_dir, train_time)
