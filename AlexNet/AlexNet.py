import numpy as np
import torch
import torchvision
import matplotlib as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

NUM_CLASS = 10 #클래스 개수. fashionMnist의 경우 클래스가 10개
NUM_EPOCH = 90 #논문에서는 90
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tensorboard_logs'

transform = transforms.Compose([
    transforms.Resize(227), #논문에는 224*224 이미지가 입력으로 들어오나, 224로 할 경우 아웃풋 사이즈가 55가 될 수 없음. (227-11)/4+1 = 55. 224는 오류이며, 227이 맞다고 논문 저자가 수정함.
    transforms.ToTensor()
])

#논문에서는 ImageNet 데이터를 사용하나, 용량이 너무 커 fashinMnist로 대체한다.
training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transform
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = transform    
)


training_data_loader = DataLoader(training_data, batch_size=128, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=128, shuffle=True)

class AlexNet_fashionMnist(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        # 입력 이미지 : batch_size*3*227*227
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4), #출력 : (227-11)/4+1=55,   batch_size*96*55*55
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), #출력 : (55-3)/2+1=27,   batch_size*96*27*27
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), #출력 : (27-5+2*2)/1+1=27,   batch_size*256*27*27
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), #출력 : (27-3)/2+1=13,   batch_size*256*13*13
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), #출력 : (13-3+2*1)/1+1=13,   batch_size*384*13*13
            nn.ReLU(),
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), #출력 : (13-3+2*1)/1+1=13,   batch_size*384*13*13
            nn.ReLU(),
                        
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), #출력 : (13-3+2*1)/1+1=13,   batch_size*256*13*13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #출력 : (13-3)/2+1=6,   batch_size*256*6*6   
        )
        
        
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            
            nn.Linear(in_features=4096, out_features=NUM_CLASS)                        
        )
        
        self.init_bias() #바이어스 초기값 설정
        
    def init_bias(self): #논문에서 모든 weight 초기값은 mean=0, std=0.01로 설정. bias는 2,4,5번째 convolution레이어만 1로 설정
        for layer in self.net:
            if isinstance(layer, nn.Conv2d): 
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
                
        nn.init.constant_(self.net[4].bias, 1) #2
        nn.init.constant_(self.net[10].bias, 1) #4
        nn.init.constant_(self.net[12].bias, 1) #5

    def forward(self,x):
        x = self.net(x)
        x = x.view(-1, 256*6*6) #flatten
        return self.linear(x)
    

if __name__ == '__main__': #해당 스크립트를 직접 실행할 때만 실행됨.
    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    alexnet = AlexNet_fashionMnist(num_class = NUM_CLASS).to(device)    
    #print(alexnet)
    print('AlexNet created')
    
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    # 논문에서는 optimer로 SGD를 사용함.
    #   optimizer = optim.SGD(
    #   params=alexnet.parameters(),
    #   lr=LR_INIT,
    #   momentum=MOMENTUM,
    #   weight_decay=LR_DECAY)
    print('Optimizer created')

    criterion = F.cross_entropy


    def train(model, device, training_data_loader, optimizer, epoch):
        model.train()
        
        for (batch_idx), (imgs, classes) in enumerate(training_data_loader):
            imgs, classes  = imgs.to(device), classes.to(device)
            
            
            # 손실함수 계산
            output = alexnet(imgs)
            loss = criterion(output, classes)
            
            # 파라미터 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if(batch_idx % 50 == 0):
                print("Train Epoch: {}  [{}/{}]\t({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx*len(imgs), len(training_data_loader.dataset), 100.*batch_idx/len(training_data_loader), loss.item()))
                


    def test(model, device, test_data_loader):
        model.eval()
        test_loss = 0
        correct = 0  
        with torch.no_grad():
            for imgs, classes in test_data_loader:
                imgs, classes  = imgs.to(device), classes.to(device)
                output = alexnet(imgs)
                test_loss += criterion(output, classes, reduction='sum').item()
                pred = F.softmax(output, dim=1)  # softmax를 적용하여 확률값으로 변환
                pred_class = pred.argmax(dim=1, keepdim=True)  # 가장 높은 확률을 가진 클래스의 인덱스 추출
                correct += pred_class.eq(classes.view_as(pred_class)).sum().item()


        test_loss /= len(test_data_loader.dataset)  # -> batch로 넣었기 때문에 평균을 구함.
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_data_loader.dataset), 100. * correct / len(test_data_loader.dataset)))
        print('='*50)
        
    for epoch in range(1, NUM_EPOCH+1):
        train(alexnet, device, training_data_loader, optimizer, epoch)
        test(alexnet, device, test_data_loader)




######################################################################## 아래 코드는 summarywriter를 연습해보기 위해 구현해 본 것으로 논문과는 상관 없음###############################################################################
    # def train(model, device, training_data_loader, optimizer):
    #     total_step=1
    #     for epoch in range(NUM_EPOCH):
    #         model.train()
    #         for (batch_idx), (imgs, classes) in enumerate(training_data_loader):
    #             imgs, classes  = imgs.to(device), classes.to(device)
                
                
    #             # 손실함수 계산
    #             output = alexnet(imgs)
    #             loss = criterion(output, classes)
                
    #             # 파라미터 업데이트
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    
                
    #             if(batch_idx % 50 == 0):
    #                 with torch.no_grad():
    #                     _, preds = torch.max(output, 1)
    #                     accuracy = torch.sum(preds == classes)

    #                     print('Epoch: {} \tBatchidx: {} \tLoss: {:.4f} \tAcc: {}'
    #                         .format(epoch + 1, batch_idx, loss.item(), accuracy.item()))
    #                     tbwriter.add_scalar('loss', loss.item(), batch_idx)
    #                     tbwriter.add_scalar('accuracy', accuracy.item(), batch_idx)
        


    # train(alexnet, device, training_data_loader, optimizer)
