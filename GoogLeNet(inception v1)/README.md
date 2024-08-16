# GoogleNet 구현 (CIFAR-10)

이 저장소는 **GoogleNet** 아키텍처를 이미지 분류에 사용한 구현을 포함하고 있습니다. 이 구현은 Christian Szegedy 외의 논문 "[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)"을 참고하여 CIFAR-10 데이터셋을 사용하여 구현되었습니다.

## 개요

GoogleNet은 깊이 있는 신경망 구조를 통해 이미지 분류의 성능을 크게 향상시킨 모델입니다. 이 구현은 원래 GoogleNet 모델을 CIFAR-10 데이터셋에 맞게 조정하여 사용합니다. CIFAR-10은 32x32 픽셀의 컬러 이미지로 구성된 데이터셋이며, 10개의 서로 다른 클래스가 포함되어 있습니다.

## 주요 기능

- **딥 컨볼루션 신경망**: GoogleNet 아키텍처를 기반으로 하여 CIFAR-10에 맞게 조정되었습니다.
- **CIFAR-10 데이터셋**: 32x32 픽셀 컬러 이미지로 구성된 데이터셋이며, 10개의 클래스를 포함합니다.


## 필수 사항

코드를 실행하기 전에 다음 의존성을 설치해 주세요:

- Python 3.x
- PyTorch
- NumPy
- torch
- torchvision
- torchsummary
- copy
- time
- os
- [CIFAR-10 데이터셋](https://www.cs.toronto.edu/~kriz/cifar.html) (스크립트 실행 시 자동으로 다운로드됨)

## 코드 실행 방법

모델 학습을 시작하려면 다음 명령어를 사용하세요:

```bash
python GoogleNet.py

