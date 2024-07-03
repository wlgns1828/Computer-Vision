import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as utils
from torchsummary import summary
import copy
import time
import os


#ImageNet 데이터는 용량이 너무 커 STL10으로 대체함.
train_data = datasets.STL10('GoogLeNet_img_data', download=True, split='train', transform=transforms.ToTensor())
test_data = datasets.STL10('GoogLeNet_img_data', download=True, split='test', transform=transforms.ToTensor())

print(len(train_data))
print(len(test_data))

train_meanRGB = [np.mean(x.numpy(), axis=(1,2))for x, _ in train_data] # 평균 : train_data에서 사진 한장을 가져옴. (axis = 0)는 채널, (axis = 1,2)는 가로와 세로.
train_stdRGB = [np.std(x.numpy(), axis=(1,2))for x, _ in train_data]# 표준편차
#각 채널의 RGB평균을 구함. [Rmean, Gmean, Bmean] 백터가 6000개 쌓여 있는 형태.


train_meanR = np.mean([c[0] for c in train_meanRGB]) #Rmean 6000개의 평균
train_meanG = np.mean([c[1] for c in train_meanRGB]) #Gmean 6000개의 평균
train_meanB = np.mean([c[2] for c in train_meanRGB]) #Bmean 6000개의 평균

train_stdR = np.mean([s[0] for s in train_stdRGB]) #표준편차도 동일하게 계산
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])


#테스트 데이터도 동일한 과정 수행
test_meanRGB = [np.mean(x.numpy(), axis=(1,2))for x, _ in test_data]
test_stdRGB = [np.std(x.numpy(), axis=(1,2))for x, _ in test_data]

test_meanR = np.mean([c[0] for c in test_meanRGB])
test_meanG = np.mean([c[1] for c in test_meanRGB])
test_meanB = np.mean([c[2] for c in test_meanRGB])

test_stdR = np.mean([s[0] for s in test_stdRGB])
test_stdG = np.mean([s[1] for s in test_stdRGB])
test_stdB = np.mean([s[2] for s in test_stdRGB])

print(train_meanR, train_meanG, train_meanB)
print(test_meanR, test_meanG, test_meanB)


#이미지를 텐서형태로 바꾸고, 224사이즈로 변형. 위에서 구한 평균과 표준편차로 정규화 한 뒤, 좌우대칭 랜덤으로 가져옴.
train_transformation = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(224),
                                           transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
                                           transforms.RandomHorizontalFlip()
])

test_transformation = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(224),
                                           transforms.Normalize([test_meanR, test_meanG, test_meanB], [test_stdR, test_stdG, test_stdB]),                                           
])
train_data.transform = train_transformation
test_data.transform = test_transformation

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True)


#################################################################################################################################################

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=10, init_weights = True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)#3*224*224 > 64*112*112.
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 64*56*56
        
        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1) # 64*56*56 > 192*56*56
        self.relu2 = nn.ReLU()
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 192*28*28
        
        
        self.inception3a = Inception_block(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32) #4개의 branch를 합치면 총 256채널이 출력됨. 이미지 사이즈는 변하지 않음.
        self.inception3b = Inception_block(in_channels=256, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64) #4개의 branch를 합치면 총 480채널이 출력됨. 480*28*28
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #480*14*14
        
        self.inception4a = Inception_block(in_channels=480, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64) #512*14*14

        # auxiliary classifier
        self.inception4b = Inception_block(in_channels=512, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64) #512*14*14
        self.inception4c = Inception_block(in_channels=512, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64) #512*14*14
        self.inception4d = Inception_block(in_channels=512, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64) #528*14*14

        # auxiliary classifier
        self.inception4e = Inception_block(in_channels=528, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128) #832*14*14
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #832*7*7
        
        self.inception5a = Inception_block(in_channels=832, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128) #832*7*7
        self.inception5b = Inception_block(in_channels=832, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128) #1024*7*7

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1) #832*1*1
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes) #1000*1*1

        if self.aux_logits: #훈련일 때 만 사용
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

        # 가중치 초기화
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = x.view(x.shape[0], -1)

        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x 

    # 가중치 초기화 함수
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) #논문에서 초기화 기법에 대한 언급이 없어 임의로 xavier로 설정함.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)            
        )
    
    def forward(self, x):
        return self.conv_layer(x)


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # 0차원은 batch이므로 1차원인 filter 수를 기준으로 각 branch의 출력값을 묶음. 
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x

# auxiliary classifier의 loss는 0.3이 곱해지고, 최종 loss에 추가됨. 정규화 효과가 있다. 
class InceptionAux(nn.Module):#Aux는 GoogLeNet에서 2번 사용됨. 첫번째 : (512*14*14), 두번째 : (518*14*14). in_channels만 다르고 사이즈는 동일함.
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3), #14*14 > 4*4
            conv_block(in_channels, out_channels=128, kernel_size=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),#128*4*4=2048
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = GoogLeNet(aux_logits=True, num_classes=10, init_weights=True).to(device)
summary(model, input_size=(3,224,224), device=device.type)

######################################################################################################################################

loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)


from torch.optim.lr_scheduler import StepLR
lr_scheduler = StepLR(opt, step_size=30, gamma=0.1) #30epoch마다 lr을 lr*0.1만큼 감소시킴

def get_lr(opt):#현재 lr을 출력하는 함수
    for param_group in opt.param_groups:
        return param_group['lr']
    
def metric_batch(output, target):#맞춘  개수 출력
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


def loss_batch(loss_func, outputs, target, opt=None):
#GoogLeNet은 출력이 총 3개가 나오므로 3개의 loss를 더함. main_loss + 0.3*(aux1_loss+aux2_loss). batch_size만큼 한번에 실행됨
    if isinstance(outputs, tuple) and len(outputs) == 3:#보조 출력기를 사용할 경우만 실행한다.
        output, aux1, aux2 = outputs

        output_loss = loss_func(output, target)
        aux1_loss = loss_func(aux1, target)
        aux2_loss = loss_func(aux2, target)

        loss = output_loss + 0.3*(aux1_loss + aux2_loss)
        metric_b = metric_batch(output,target)

    else:
        loss = loss_func(outputs, target) # 보조 출력기를 사용하지 않을경우 인자를 나눌 필요 없이 바로 계산
        metric_b = metric_batch(outputs, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):#
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output= model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b #배치마 loss를 계산하기 때문에 모든 배치의 loss를 더해줌.

        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric


def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_data_loader=params["train_data_loader"]
    test_data_loader=params["test_data_loader"]
    sanity_check=params["sanity_check"] #True일 경우 1epoch만 수행함.
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history = {'train': [], 'test': []}
    metric_history = {'train': [], 'test': []}

    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')
    
    start_time = time.time()
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_data_loader, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            test_loss, test_metric = loss_epoch(model, loss_func, test_data_loader, sanity_check)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')

        loss_history['test'].append(test_loss)
        metric_history['test'].append(test_metric)

        lr_scheduler.step()

        print('train loss: %.6f, test loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, test_loss, 100*test_metric, (time.time()-start_time)/60))
        print('-'*10)

    model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history

params_train = {
    'num_epochs':100,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_data_loader':train_data_loader,
    'test_data_loader':test_data_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSerror:
        print('Error')
createFolder('./models')

model, loss_hist, metric_hist = train_val(model, params_train)

num_epochs=params_train["num_epochs"]

# plot loss progress
plt.title("Train-Test Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["test"],label="test")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Train-Test Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["test"],label="test")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

