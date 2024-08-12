본 코드는 ResNet을 구현해보기 위해 구조를 하나하나 코드로 작성하였습니다.
코드를 실행할 시 아래 내용에 맞게 수정이 필요합니다.(디렉토리 위치)


총 5개의 파일이 존재합니다.

1. load_dataset.py
2. model.py
3. train.py
4. main.py
5. graph.py


## load_dataset.py
  데이터셋을 준비하는 코드 입니다. 기본적으로 cifar10 파일을 다운로드하고 데이터로더로 불러옵니다.
  cifar10 데이터셋을 저장할 위치인 path2data를 본인의 디렉토리 위치에 맞게 수정해주세요.

## model.py
  모델 사이즈에 따라 구조가 코드로 구현되어있습니다. 수정할 부분은 없습니다.
  ResNet의 구조를 자세히 알고 싶다면 이 코드를 공부해주세요.

## train.py
  모델의 학습과정에 대한 코드입니다. 학습에 필요한 함수들만 구현한 것이기 때문에 수정할 부분은 없습니다.
  학습의 결과를 저장하는 코드도 작성되어 있습니다. training, test의 loss, accuracy, 학습시간을 저장하는 코드가 구현되어 있습니다.
  학습을 하고 싶을 때 train.py를 실행하는 것이 아닌 main.py를 실행해야 합니다.

## main.py
  학습을 하고 싶을 경우 main.py를 실행해주시면 됩니다.

  수정사항:
    1. model = resnet18().to(device)
      ResNet은 사이즈가 여러개 존재합니다. 기본적으로 18개의 레이어로 학습을 하도록 구현되어 있습니다.
      다른 사이즈로 학습을 하고 싶은 경우 위 코드에서 18을 34, 50, 101, 152 중 원하는 것으로 바꿔주시면 됩니다.
    
    2. loss_func = nn.CrossEntropyLoss(reduction='sum')
       opt = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
       위 코드는 학습시 사용될 손실함수와 옵티마이저 입니다. 필요에 따라 수정해주세요.


  python main.py
  
  위 코드를 입력하면 학습을 시작합니다. models 디렉토리에 최종 weight가 저장됩니다.
  training_history18(cifar10) 디렉토리에 train과 val의 loss, accuracy, 학습시 소요된 시간이 csv파일로 각각 저장됩니다.

  ## graph.py
    학습 결과를 그래프로 보여주는 코드입니다. csv 파일에 저장된 train, val의 loss, accuracy를 그래프로 보여줍니다.
