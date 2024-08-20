from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from set_parameter import get_parser
import numpy as np

parser = get_parser()

training_data = datasets.STL10(
    root = '../../dataset',
    split = 'train',
    download = True,
    transform = transforms.ToTensor()
)

test_data = datasets.STL10(
    root = '../../dataset',
    split = 'test',
    download = True,
    transform = transforms.ToTensor()
)


train_meanRGB = [np.mean(x.numpy(), axis=(1,2))for x, _ in training_data] # 평균 : train_data에서 사진 한장을 가져옴. (axis = 0)는 채널, (axis = 1,2)는 가로와 세로.
train_stdRGB = [np.std(x.numpy(), axis=(1,2))for x, _ in training_data]# 표준편차
#각 채널의 RGB평균을 구함. [Rmean, Gmean, Bmean] 백터가 6000개 쌓여 있는 형태.


train_meanR = np.mean([c[0] for c in train_meanRGB]) #Rmean 6000개의 평균
train_meanG = np.mean([c[1] for c in train_meanRGB]) #Gmean 6000개의 평균
train_meanB = np.mean([c[2] for c in train_meanRGB]) #Bmean 6000개의 평균

train_stdR = np.mean([s[0] for s in train_stdRGB]) #표준편차도 동일하게 계산
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])


transform = transforms.Compose([
    transforms.Resize(parser.img_resize),
    transforms.ToTensor(),
    transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB])
])

training_data.transform = transform
test_data.transform = transform

training_data_loader = DataLoader(training_data, batch_size=parser.batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=parser.batch_size, shuffle=True)

