import os 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 

def create_loaders(batch_size=100):
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transforms.ToTensor(), 
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    return train_loader, test_loader


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 500) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)  
    
    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_model(input_size=784, num_classes=10):   # 784 = 28x28
    global device
    m = NeuralNet(input_size, num_classes)
    
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f'running on GPU. number og GPUs: {count}')
        for i in range(count):
            print(f'#{i} name : {torch.cuda.get_device_name(i)}')
        device = 'cuda'
    else:
        print('Running on CPU')
        device = 'cpu'
    return m.to(device)


def train(model, train_loader, num_epochs=5, learning_rate=1e-3):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            images = images.to(device) 
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')


def test(model, test_loader):
    model.eval()  # Evaluation mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the {len(test_loader)} test images: {100 * correct / total} %')


def save(model,filename='model.ckpt'):
    torch.save(model.state_dict(), filename)


def load_model(filename='model.ckpt'):
    model = create_model()
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    return model 


def my_digits_test():
    dig_folder = './digits/'
    filenames = os.listdir(dig_folder)
    model = load_model()

    for filename in filenames: 
        img = plt.imread(dig_folder + filename)
        plt.imshow(img)
        plt.show()

        t = torch.Tensor(img).to(device)
        t = t.reshape(-1, 28*28)
        p = model(t)
        # print(f'{p.shape}   {p}')
        _, c = torch.max(p.data, 1)
        print(f'{filename} --> {c}')


if __name__ == '__main__':
    # train_loader, test_loader = create_loaders()
    # model = create_model()
    # train(model, train_loader)
    # test(model, test_loader)
    # save(model)    
    my_digits_test()
