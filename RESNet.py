import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import torch
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms

batch_size_train = 64
batch_size_test = 1000


dataset_train =  torchvision.datasets.MNIST('/home/alessio/alonso/datasets', train=True, download=True,
                             transform=torchvision.transforms.ToTensor())

dataset_test =  torchvision.datasets.MNIST('/home/alessio/alonso/datasets', train=False, 
                                           download=True,transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train,drop_last=True, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size_test, shuffle = True,drop_last=True)

shuffled_train = Shuffdata.ShuffleMNIST(train_loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True,is_train=True)
shuffled_test = Shuffdata.ShuffleMNIST(test_loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True, is_train = False)

print('There are {} images and {} labels in the train set.'.format(len(shuffled_train.train_img),
        len(shuffled_train.train_label)))
print('There are {} images and {} labels in the test set.'.format(len(shuffled_test.test_img),
        len(shuffled_test.test_label)))

#Configuring shuffled DataLoader
from torch.utils.data.sampler import RandomSampler

#se cambian estos nombres a train loader para que sean los que se llaman en la red
train_sampler = RandomSampler(shuffled_train, replacement=True, num_samples= 51200, generator=None)
test_sampler = RandomSampler(shuffled_test, replacement=True, num_samples= 5760, generator=None)

trainshuffled_loader = torch.utils.data.DataLoader(shuffled_train, batch_size=batch_size_train
                                                   ,drop_last=False, sampler = train_sampler)

testshuffled_loader = torch.utils.data.DataLoader(shuffled_test, batch_size=batch_size_train
                                                  ,drop_last=False, sampler = test_sampler)


resnet18 = models.resnet18()

#configurando la led para las targetas gráficas
net = models.resnet18(pretrained=True)
net = net.cuda()
net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 37)
net.fc = net.fc.cuda()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#para el nombre de las imágenes
count_fig = 0

n_epochs = 100
print_every = 100
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(trainshuffled_loader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(trainshuffled_loader):
        data_, target_ = data_, target_.to(device)
        optimizer.zero_grad()

        img = np.array(data_)
        #print(img.shape)
        if len(img.shape) == 3:
                img = np.stack([img] * 3, 2)

        
        img = np.transpose(img, (0,2,1,3))
        img = torch.as_tensor(img)

        #print(img.shape)
        data_ = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        data_ = data_.to(device)
        
        outputs = net(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (testshuffled_loader):
            data_t, target_t = data_t, target_t.to(device)

            #transformacion de los datos
            img = np.array(data_t)
            #print(img.shape)
            if len(img.shape) == 3:
                    img = np.stack([img] * 3, 2)

            
            img = np.transpose(img, (0,2,1,3))
            img = torch.as_tensor(img)

            #print(img.shape)
            data_t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            data_t = data_.to(device)


            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(testshuffled_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        #Queremos graficar el entrenamiento
        #Puede esto verse a 'tiempo real'??
        #mientras lo graficarmeos cada 100 epocs


        count_fig =+ 1
        fig = plt.figure(figsize=(20,10))
        plt.title("Train-Validation Accuracy")
        plt.plot(train_acc, label='train')
        plt.plot(val_acc, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(f'prueba{count_fig}.png')

        
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    net.train()