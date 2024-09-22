import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ClassifyNet(nn.Module):
    def __init__(self):
        super(ClassifyNet, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def eval(model, test_data):
    with torch.no_grad():
        inputs = torch.from_numpy(test_data["X"]).float()
        labels = torch.from_numpy(test_data["Y"]).long()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        return acc.item()

def vis_loss(losses, acc_reord):
    # plt.title("loss ", loc = "center")

    figure, (ax1, ax2) = plt.subplots(1,2)

    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss ")
    ax1.plot(losses)

    ax2.set_title("accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("acc ")
    ax2.plot(acc_reord)
    plt.savefig("./loss.png")
    # plt.show()

def train(train_data, test_data):
    model = ClassifyNet()
    Loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.90)

    losses_record = []
    acc_record = []
    num_epochs = 150
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(train_data["X"]).float()
        labels = torch.from_numpy(train_data["Y"]).long()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = Loss(outputs, labels)
        loss.backward()
        optimizer.step()
        losses_record.append(loss.item())
        if(epoch % 10 == 0):
            print("Epoch: %d, Loss: %.4f" %(epoch, loss.item()))
        
        acc = eval(model, test_data)
        acc_record.append(acc)

    vis_loss(losses_record, acc_record)

    acc = eval(model, test_data)
    print("Accuracy: %.2f %% "%(acc * 100))
if __name__ == "__main__":
    np.random.seed(10)

    dat = pd.read_csv("iris.csv", header=None)
    # dat.iloc[:, 0:4].mean()
    X = dat.iloc[:, 0:4].values
    dat.iloc[:, 4].replace(("setosa", "versicolor", "virginica"), (0, 1, 2), inplace=True)
    labels = dat.iloc[:, 4]
    
    X = np.array(X)
    labels = np.array(labels)
    data  = np.zeros((len(X), X.shape[1] + 1))
    data[:, :4] = X
    data[:, 4:5] = labels.reshape((-1, 1))
    np.random.shuffle(data)

    # split train and test

    train_data_arr = data[:120]
    test_data_arr = data[120:150]

    train_data = {"X":train_data_arr[:, :4], "Y":train_data_arr[:, 4:5].reshape((-1, ))}
    test_data = {"X":test_data_arr[:, :4], "Y":test_data_arr[:, 4:5].reshape((-1, ))}

    train(train_data, test_data)
