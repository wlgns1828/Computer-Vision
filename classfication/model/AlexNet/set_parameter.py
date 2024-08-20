import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--n_class", type=int, default=10, help="number of class")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_resize", type=int, default=227, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--result_path", type=str, default='./result', help="path of result")
    
    parsers = parser.parse_args()
    return parsers