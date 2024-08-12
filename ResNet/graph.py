import os
import matplotlib.pyplot as plt
import csv

# from main import get_params_train
# params=get_params_train()
# num_epochs=params['num_epochs']

num_epochs = 50

folder_path = './training_history'
file_names = os.listdir(folder_path)

loss_hist = {"train": [], "val": []}
metric_hist = {"train": [], "val": []}

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 첫 번째 행(헤더) 건너뛰기
        data = []
        for row in reader:
            data.append(float(row[0]))

    if 'train_loss' in file_name:
        loss_hist["train"] = data
    elif 'val_loss' in file_name:
        loss_hist["val"] = data
    elif 'train_metric' in file_name:
        metric_hist["train"] = data
    elif 'val_metric' in file_name:
        metric_hist["val"] = data


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
plt.title("Train and Validation Loss")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
plt.plot(range(1, num_epochs + 1), metric_hist["val"], label="val")
plt.title("Train and Validation Metric")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()