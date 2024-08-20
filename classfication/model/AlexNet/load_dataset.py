from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from set_parameter import get_parser

parser = get_parser()

transform = transforms.Compose([
    transforms.Resize(parser.img_resize), #논문에는 224*224 이미지가 입력으로 들어오나, 224로 할 경우 아웃풋 사이즈가 55가 될 수 없음. (227-11)/4+1 = 55. 224는 오류이며, 227이 맞다고 논문 저자가 수정함.
    transforms.ToTensor()
])

training_data = datasets.FashionMNIST(
    root = '../../dataset',
    train = True,
    download = True,
    transform = transform
)

test_data = datasets.FashionMNIST(
    root = '../../dataset',
    train = False,
    download = True,
    transform = transform    
)

training_data_loader = DataLoader(training_data, batch_size=parser.batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=parser.batch_size, shuffle=True)