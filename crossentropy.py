import pandas as pd
import numpy as np
import conf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class ValueNN(nn.Module):
    def __init__(self):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(conf.FEATURE_NUMB, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, conf.ACTIONS_NUMB)

    def forward(self, status):
        status = F.relu(self.fc1(status))
        status = F.relu(self.fc2(status))
        status = self.fc3(status)
        return status


class MovementDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset.values

    def __getitem__(self, item):
        return torch.Tensor(np.asarray(self.dataset[item, :-1])), torch.LongTensor(np.asarray(self.dataset[item, -1]))

    def __len__(self):
        return self.dataset.shape[0]


####################
#   MAIN
####################


dataset = pd.read_csv(conf.DATASET_PATH)
# normalizing..
norm = dataset.copy()
for feature_name in dataset.columns[:-1]:
    max_value = dataset[feature_name].max()
    min_value = dataset[feature_name].min()
    norm[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)
dataset = norm
# splitting dataset for training and testing
train, test = train_test_split(dataset, test_size=0.2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# creating iterator for training test
trainset = MovementDataset(train)
train_loader = DataLoader(trainset, batch_size=conf.BATCH_SIZE, shuffle=True)

# creating the network and its stuff
net = ValueNN()
net.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=conf.LEARNING_RATE)

net.train()

for k in range(conf.EPOCH):
    for step, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        predictions = net(x)

        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        if conf.DEBUG:
            print('epoch: ', k, step, ' loss for this batch: ', loss.item())


net.eval()
testset = MovementDataset(test)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
positive = 0
for step, data in enumerate(test_loader):
    x, y = data
    x = x.to(device)
    y = y.to(device)
    predictions = net(x)
    predicted = torch.argmax(predictions, dim=1)
    print(int(predicted))
    if predicted == y:
        positive += 1

print("guessed: {}, total: {}".format(positive, len(test.index)))
print("accuracy on test set: {}%".format(round(100*positive/len(test.index)), 2))
print("save weights? (y | enter to skip)")
save = input()
if save == 'y':
    torch.save(net.state_dict(), 'weights.dat')
