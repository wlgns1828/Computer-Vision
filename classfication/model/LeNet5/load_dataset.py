from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from set_parameter import get_parser

parser = get_parser()

transform = transforms.Compose([
    transforms.Resize(parser.img_resize),
    transforms.ToTensor()
])

training_data = datasets.MNIST(
    root = '../../dataset',
    train = True,
    download = True,
    transform = transform
)

test_data = datasets.MNIST(
    root = '../../dataset',
    train = False,
    download = True,
    transform = transform    
)

training_data_loader = DataLoader(training_data, batch_size=parser.batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=parser.batch_size, shuffle=True)