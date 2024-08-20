# 📌 Computer VIsion Research Code

이 저장소는 다양한 컴퓨터 비전 논문을 구현한 코드가 저장되어있습니다. 

이미지 분류를 위한 다양한 모델을 포함하고 있으며, 각 모델은 학습 및 평가가 가능합니다.
각 에포크 마다 loss와 accuracy를 csv 파일에 저장하며, 그래프를 그려 png 형식으로 저장할 수 있도록 구현하였습니다.

가능한 원하는 용도에 따라 수정이 편하도록 코드를 나눠 구현하였습니다.
파라미터 값을 수정하기 용이하며, 모델의 구조를 공부하기 편하도록 훈련 코드와 분리하여 작성하였습니다.


## 📢 목차

1. [프로젝트 구조](#프로젝트-구조)

2. [파일 설명](#파일-설명)
    - [load_dataset.py](#load_dataset.py)
    - [set_parameter.py](#set_parameter.py)
    - [model.py](#model.py)
    - [train.py](#train.py)
    - [graph.py](#graph.py)
3. [훈련 방법](#훈련-방법)

-----

## 👉 프로젝트 구조

- classification
    - dataset

    - model
        - LeNet5
        - AlexNet
        - GoogLeNet
        - VGGNet
        - ResNet

각 모델 이름이 적힌 폴더 안에는 아래 파일들이 들어 있습니다.

- load_dataset.py
- set_parameter.py
- model.py
- train.py
- grapy.ph

-----

## 👉 파일 설명

### 1. load_dataset.py

    훈련에 사용할 이미지를 다운로드 하고 DataLoader로 불러옵니다.
    Mnist, FashionMnist, Cifar10 데이터셋 등 모델마다 데이터셋이 다릅니다.

    훈련에 사용할 데이터셋을 변경하고 싶으시다면 해당 파일의 코드를 수정해주세요.

### 2. set_parameter.py

    훈련에 사용되는 파라미터 값을 정합니다.
    learning rate, num_epoch, num_class 등 필요따라 수정해주세요.

### 3. model.py

    논문에서 소개한 모델을 구현한 코드 입니다.
    해당 논문을 공부하고 싶다면 이 파일을 중심적으로 봐주세요.

### 4. train.py

    훈련 과정이 작성된 파일입니다.
    훈련 과정이나, 결과를 저장하는 방식을 변경하고 싶다면 해당 코드를 수정해주세요.

### 5. graph.py

    훈련 결과를 저장하고 그래프로 표현하는 함수가 저장된 파일입니다.
    epoch 마다 loss와 accuracy를 csv 파일에 저장합니다.
    csv 파일을 이용해 그래프를 그려 시각화 합니다.
    result 파일에 그래프를 png 형식으로 저장합니다.

![training_results](https://github.com/user-attachments/assets/6edc9b4b-1883-4e8b-b7d7-4d986d2eec56)


-----

## 📌 훈련 방법

깃허브에서 코드를 가져옵니다.

> git.clone ~~

<br>

필요한 패키지를 install 합니다.

pip install -r requirements.txt

<br>

디렉토리 위치를 훈련하고 싶은 모델로 이동합니다.

> cd ./classification/model/AlexNet

<br>

예시로 AlexNet 폴더로 이동했습니다.

훈련을 시작합니다.

<br>

> python train.py

훈련결과 result 폴더가 생성되며 csv파일과, 훈련 결과 그래프가 png파일로 저장됩니다.

