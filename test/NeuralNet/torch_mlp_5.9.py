import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

class MLP(nn.Module):
    '''
    -------- Multilayer Perceptron. --------
    '''
    def __init__(self):
        super().__init__()
        input_feature = 4
        hidden_1 = 15
        hidden_2 = 3
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
    path = '../../../../Documentos/Mestrado_UTFPR/Disciplinas/2020_02/Machine_Learning/Dados/C/5.9/'
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
    data_train = np.loadtxt(path + 'Data.txt')
    data_train = data_train[:,1:]
    #bias_inp =np.ones((len(data_train),1))

    #data_train = np.concatenate((bias_inp,data_train), axis=1)
    data_train = torch.tensor(data_train)

    data_train = data_train.to(device)
    # Initialize the MLP 
    # Set double to float64 and to gpu
    #torch.manual_seed(100)
    mlp = MLP().double().to(device)

    # Define the loss function and optimizer
    #loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1, momentum=0.0)
    #optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    # Run the training loop

    # %% Configuration

    n_epoch = 1000
    '''
     -------- Training the model --------
    '''
    MSE = np.zeros((n_epoch,))
    aux_mse = []
    mse_t = 0.0
    loss_m = 0.0
    for epoch in range(0, n_epoch): # 5 epochs at maximum
        # Print epoch
        #print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0
        mse = 0.0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(data_train, 0):
            # Get inputs
            inputs = data[:-3]
            targets = torch.reshape(data[4:],(3,))

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
            current_loss += 0.5*loss.item()
            mse += 0.5*np.sum((targets.cpu().data.numpy()-outputs.cpu().data.numpy())**2)
            #mse += 0.5*np.sum((targets-outputs)**2)
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0
        teste = np.abs(loss_m-current_loss/len(data_train))
        #print('{} \t {}'.format(teste,
        #        np.abs(mse_t-mse/len(data_train))))
        print('{}\t{}'.format(epoch, teste))
        loss_m = current_loss/len(data_train)
        #print('{} \t {} \t {}'.format(mse/len(data_train),current_loss/len(data_train),loss.data.item()))
        mse_t = mse/len(data_train)
        aux_mse.append(mse/len(data_train))
        if((10**6)*teste < 1):
            break
    plt.plot(aux_mse)
    plt.show()
    '''
    aux = loss.data.item()
        print('{}  {}'.format(aux, current_loss/len(data_train)))
        if (np.abs(aux-mse) < 10**-6):
            print("Eu")
            break
        else:
            continue
        mse = aux
        MSE[epoch] = mse
    # Process is complete.
    print('Training process has finished.')
    print(epoch)
    #aux2 = np.argwhere(MSE)
    plt.plot(MSE)
    plt.show()
'''
    # %% Validate
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
    out = np.zeros((len(data_test),3))
    for i, data in enumerate(data_test, 0):
        # forward pass: compute predicted outputs by passing inputs to the model
        inputs = data[:-3]
        targets = torch.reshape(data[4:],(3,))

        output = mlp(inputs)
        out[i, :] = output.cpu().data.numpy()
        # calculate the loss
        loss = loss_function(output, targets)
        # update running validation loss
        #valid_loss += loss.item()*data.size(0)
    out_pos = np.where(out>=.5, 1, 0)
    print(out_pos)
    alvo = data_test[:,4:].cpu().data.numpy()
    print(out_pos-alvo)

