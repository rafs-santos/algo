import torch
from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np

class MLP(nn.Module):
    '''
    -------- Multilayer Perceptron. --------
    '''
    def __init__(self):
        super().__init__()
        input_feature = 3
        hidden_1 = 10
        hidden_2 = 1
        self.layers = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(input_feature, hidden_1),
            nn.Sigmoid(),
            nn.Linear(hidden_1, hidden_2, bias=True),
            nn.Sigmoid()
        )
        print(self.layers)
    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

if __name__ == '__main__':
    path = '../../../../Documentos/Mestrado_UTFPR/Disciplinas/2020_02/Machine_Learning/Dados/C/5.8/'
    '''
    Trying put data in gpu
    '''
    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"
    device = torch.device(dev)
    '''
     -------- Load Data to Traning --------
    '''
    data_train = np.loadtxt(path + 'Pasta1.txt')
    data_train = data_train[:,1:]
    #bias_inp =np.ones((len(data_train),1))

    #data_train = np.concatenate((bias_inp,data_train), axis=1)
    data_train = torch.tensor(data_train)

    data_train = data_train.to(device)
    # Initialize the MLP 
    # Set double to float64 and to gpu
    mlp = MLP().double().to(device)

    # Define the loss function and optimizer
    #loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    #optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    # Run the training loop

    # %% Configuration

    n_epoch = 50
    '''
     -------- Training the model --------
    '''
    for epoch in range(0, n_epoch): # 5 epochs at maximum
        # Print epoch
        #print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(data_train, 0):
            # Get inputs
            inputs = data[:-1]
            targets = torch.reshape(data[-1],(1,))

            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = mlp(inputs)
            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')
    '''
    -------- validate the model --------
    '''
    mlp.eval() # prep model for evaluation
    data_test = np.loadtxt(path + 'Valida.txt')
    data_test = data_test[:,1:]
    #bias_inp =np.ones((len(data_test),1))

    #data_test = np.concatenate((bias_inp,data_test), axis=1)
    data_test = torch.tensor(data_test)

    data_test = data_test.to(device)
    out = np.zeros((len(data_test),1))
    for i, data in enumerate(data_test, 0):
        # forward pass: compute predicted outputs by passing inputs to the model
        inputs = data[:-1]
        targets = torch.reshape(data[-1],(1,))

        output = mlp(inputs)
        out[i, 0] = output
        # calculate the loss
        loss = loss_function(output, targets)
        # update running validation loss
        #valid_loss += loss.item()*data.size(0)
    print(out)
#x = np.random.randn(1, 10)
#x1 = torch.randn(1, 10)

# %% Teste
'''
#To open mnist dataset
#import gzip
#import pickle


print(torch.cuda.is_available())
#path = 'data/FashionMNIST/processed/test.pt
#with gzip.open('mnist.pkl.gz', 'rb') as f:
#    train_set, valid_set, test_set = pickle.load(f)

'''

'''

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

'''
