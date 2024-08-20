import torch.nn as nn
from set_parameter import get_parser
parser = get_parser()

class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        # 입력 이미지 : batch_size*3*227*227
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=parser.channels, out_channels=96, kernel_size=11, stride=4), #출력 : (227-11)/4+1=55,   batch_size*96*55*55
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
            
            nn.Linear(in_features=4096, out_features=parser.n_class)                        
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
