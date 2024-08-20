import torch.nn as nn
import torch
from set_parameter import get_parser
parser = get_parser()

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=parser.n_class, init_weights = True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)#3*224*224 > 64*112*112.
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 64*56*56
        
        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1) # 64*56*56 > 192*56*56
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

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1) #1024*1*1
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
        x = self.conv_1x1(x)
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
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
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
