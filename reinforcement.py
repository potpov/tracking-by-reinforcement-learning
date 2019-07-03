import pandas as pd
import numpy as np
import conf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys


class ValueNN(nn.Module):
    def __init__(self):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(conf.FEATURE_NUMB, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, conf.ACTIONS_NUMB)

    def forward(self, status):
        status = F.relu(self.fc1(status))
        status = F.relu(self.fc2(status))
        status = F.dropout(F.relu(self.fc3(status)), training=self.training)
        status = self.fc4(status)
        return status


class MovementDataset(Dataset):
    def __init__(self, dataset):
        # extracting the last row from the dataframe (to avoid overflow in get_item)
        self.last_row = dataset.tail(1)
        self.dataset = dataset.copy()
        self.dataset.drop(dataset.tail(1).index, inplace=True)
        # saving dataset as numpy matrix
        self.dataset = self.dataset.values

    def __getitem__(self, index):
        x = torch.Tensor(np.asarray(self.dataset[index, :-1]))
        y = torch.Tensor(np.asarray(self.dataset[index, -1]))
        if index + 1 == self.dataset.shape[0]:
            next_x = torch.Tensor(np.asarray(self.last_row)[:, :-1])
        else:
            next_x = torch.Tensor(np.asarray(self.dataset[index+1, :-1]))
        return x, y, next_x.squeeze()

    def __len__(self):
        return self.dataset.shape[0]


class Agent:
    def __init__(self, training=True, weights_path=None):
        """
        updates the value function using the Q-learning algorithm
        :param training: boolean var to split directly to test
        :param weights_path: optional path to load weights and skip training
        """
        self.net = ValueNN()
        self.net.to(device)
        if training:
            self.criterion = nn.MSELoss().to(device)
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=conf.LEARNING_RATE)
            self.net.train()
        else:  # loading weights for the network
            print("loading network from weights...")
            self.net.load_state_dict(torch.load(weights_path))
            self.test_mode()

    def test_mode(self):
        """
        Switch the network to test mode
        """
        self.net.eval()

    def get_loss_value(self):
        """
        Return the loss value for the last iteration
        :return loss value
        """
        return self.loss.item()

    def save_session(self):
        """
        Save network weights in the local folder
        """
        torch.save(self.net.state_dict(), 'weights.pth')

    def update(self, value_function, next_value_function):
        """
        updates the value function using the Q-learning algorithm
        :param value_function, chosen with e-greedy policy
        :param next_value_function, chosen with greedy policy
        """
        self.loss = self.criterion(next_value_function, value_function)
        self.loss.backward()
        self.optimizer.step()

    def e_greedy(self, x):
        """
        follows epsilon-greedy policy to choose the best action according to
        the value function or (with epsilon prob) a random integer in the actions range
        :param x: status of the agent
        :return value_function: all the actions values for this state foreach element in batch
        :return predicted: the index with the chosen action foreach element in batch
        """
        self.optimizer.zero_grad()
        value_function, predicted = self.greedy(x)
        predicted = predicted.type(torch.FloatTensor).to(device)

        # changing some prediction according to e-greedy policy for behaviour policy
        random = np.random.random_sample(predicted.size()[0])
        rand_pred = torch.Tensor(np.random.randint(0, conf.ACTIONS_NUMB, predicted.size()[0])).to(device)
        random = torch.ByteTensor(random < conf.EPSILON).to(device)  # creating a mask
        predicted = torch.where(random, rand_pred, predicted)

        return value_function, predicted

    def greedy(self, x):
        """
        follow greedy policy to choose the best action
        :param x: the current state to evaluate, can be a batch of states or a single one
        :return value_function: all the actions values for this state foreach element in batch
        :return predicted: the index with the chosen action foreach element in batch
        """
        return torch.max(self.net(x.to(device)),  dim=1)


############
#   MAIN   #
############


dataset = pd.read_csv(conf.DATASET_PATH)
# normalizing..
norm = dataset.copy()
for feature_name in dataset.columns[:-1]:
    max_value = dataset[feature_name].max()
    min_value = dataset[feature_name].min()
    norm[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)
dataset = norm

# splitting dataset for training and testing
train, test = train_test_split(dataset, test_size=0.2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# creating iterator for training test
trainset = MovementDataset(train)
train_loader = DataLoader(trainset, batch_size=conf.BATCH_SIZE, shuffle=True)

if len(sys.argv) == 1:  # no weights parameters
    # creating the agent for our mission
    agent = Agent()

    # INTO THE WILD ENVIRONMENT
    for k in range(conf.EPOCH):
        for step, data in enumerate(train_loader):
            x, y, next_x = data
            x = x.to(device)
            y = y.to(device)

            value_function, predicted = agent.e_greedy(x)
            # creating reward batch
            rewards = predicted == y  # this return an uint8 tensor
            rewards = rewards.type(torch.FloatTensor).to(device)
            rewards[rewards == 0] = -1
            # getting the next state -> target policy
            next_value_function, _ = agent.greedy(next_x)
            next_value_function = rewards + conf.GAMMA * next_value_function
            # calculating Q-learning formula
            agent.update(value_function, next_value_function)

            if conf.DEBUG:
                print('epoch: ', k, step, ' loss for this batch: ', agent.get_loss_value())

    agent.test_mode()

else:
    if len(sys.argv) > 2:
        print("USAGE: [weights_path | optional]")
        exit(-1)
    weights_path = sys.argv[1]
    agent = Agent(training=False, weights_path=weights_path)

####################
#   TEST TIME :S   #
####################


testset = MovementDataset(test)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
positive = 0
debug = []
for step, data in enumerate(test_loader):
    x, y, _ = data
    x = x.to(device)
    y = y.to(device)
    value_function, predicted = agent.greedy(x)
    predicted = predicted.type(torch.FloatTensor).to(device)
    debug.append(int(predicted))
    if predicted == y:
        positive += 1

print(debug)
print("guessed: {}, total: {}".format(positive, len(test.index)))
print("accuracy on test set: {}%".format(round(100*positive/len(test.index)), 2))
print("save weights? (y | enter to skip)")
save = input()
if save == 'y':
    agent.save_session()
