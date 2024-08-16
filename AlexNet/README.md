# AlexNet 구현 (Fashion MNIST)

이 저장소는 **AlexNet** 아키텍처 구현을 포함하고 있습니다. 이 구현은 Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton의 논문 "[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)"을 참고하여 Fashion MNIST 데이터셋을 사용하여 구현되었습니다.

## 개요

AlexNet은 이미지 분류에서 큰 돌파구를 마련한 딥러닝 모델입니다. 이 구현은 원래 AlexNet 모델을 Fashion MNIST 데이터셋에 맞게 조정하여 사용합니다. Fashion MNIST는 28x28 픽셀의 흑백 패션 아이템 이미지로 구성된 데이터셋입니다.

## 주요 기능

- **딥 컨볼루션 신경망**: AlexNet 아키텍처를 기반으로 하여 Fashion MNIST에 맞게 조정되었습니다.
- **Fashion MNIST 데이터셋**: 패션 아이템의 28x28 흑백 이미지로 구성된 데이터셋.

## 필수 사항

코드를 실행하기 전에 다음 의존성을 설치해 주세요:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib (시각화용)
- [Fashion MNIST 데이터셋](https://github.com/zalandoresearch/fashion-mnist) (스크립트 실행 시 자동으로 다운로드됨)

## 코드 실행 방법

모델 학습을 시작하려면 다음 명령어를 사용하세요:

```bash
python AlexNet.py
