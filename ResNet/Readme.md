# ResNet 구현 (CIFAR-10)

본 코드는 **ResNet**을 구현하기 위해 구조를 하나하나 코드로 작성한 예제입니다. 이 구현은 CIFAR-10 데이터셋을 사용하여 ResNet 모델을 학습시키는 코드로 구성되어 있습니다.
ResNet 논문: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

총 5개의 파일이 존재합니다:

1. `load_dataset.py`
2. `model.py`
3. `train.py`
4. `main.py`
5. `graph.py`

## 파일 설명

### `load_dataset.py`
이 파일은 데이터셋을 준비하는 코드입니다. 기본적으로 CIFAR-10 데이터를 다운로드하고 데이터로더로 불러옵니다. 

- **주의사항**: `cifar10` 데이터셋을 저장할 위치인 `path2data`를 본인의 디렉토리 위치에 맞게 수정해 주세요.

### `model.py`
이 파일은 ResNet 모델의 구조를 코드로 구현하고 있습니다. 모델 사이즈에 따라 구조가 다르게 작성되어 있으며, 현재 18개의 레이어로 학습되도록 구현되어 있습니다. ResNet의 구조를 자세히 알고 싶다면 이 코드를 참고해 주세요.

- **참고**: 모델 구조를 변경하고 싶다면, 이 파일을 수정하여 다른 사이즈의 ResNet을 구현할 수 있습니다.

### `train.py`
이 파일은 모델의 학습 과정에 대한 코드입니다. 학습에 필요한 함수들이 구현되어 있으며, 학습 결과를 저장하는 코드도 포함되어 있습니다. 

- **주의사항**: 학습을 진행하려면 `train.py`를 직접 실행하는 것이 아닌, `main.py`를 실행해 주세요.

### `main.py`
이 파일은 학습을 시작하는 주요 스크립트입니다. `main.py`를 실행하면 모델 학습이 시작됩니다.

#### 수정사항:

1. **모델 사이즈 설정**:
    ```python
    model = resnet18().to(device)
    ```
    ResNet의 사이즈를 변경하려면, 위 코드에서 `18`을 `34`, `50`, `101`, `152` 중 원하는 값으로 바꿔 주세요.

2. **손실 함수 및 옵티마이저 설정**:
    ```python
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
    ```
    손실 함수와 옵티마이저를 필요에 따라 수정해 주세요.

3. **학습 파라미터 설정**:
    ```python
    def get_params_train():
        params_train = {
            'num_epochs': 30,
            'optimizer': opt,
            'loss_func': loss_func,
            'train_dl': train_dl,
            'val_dl': val_dl,
            'sanity_check': False,
            'lr_scheduler': lr_scheduler,
            'path2weights': './models/weights_18(cifar10).pt',
        }
        return params_train
    ```
    학습 epoch 수와 `lr_scheduler` 등의 파라미터를 설정해 주세요.

### `graph.py`
이 파일은 학습 결과를 그래프로 시각화하는 코드입니다. `training_history18(cifar10)` 디렉토리에 저장된 CSV 파일에서 train 및 validation의 손실, 정확도를 그래프로 보여줍니다.

## 실행 방법

학습을 시작하려면 다음 명령어를 사용하세요:

```bash
python main.py
