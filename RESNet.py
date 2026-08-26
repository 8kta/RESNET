import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms

from ShuffleMNIST import dataset as Shuffledata
from utils import timer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('resnet.log')],
)
logger = logging.getLogger('RESNet')

logger.info('='*70)
logger.info('Experiment: GoogLeNet (from scratch) on ShuffleMNIST')
logger.info('Task: predict the SUM of 4 MNIST digits pasted on a 112x112 canvas, '
            'i.e. a 37-class classification problem (sums range from 0 to 36)')
logger.info('='*70)

batch_size_train = 64
batch_size_test = 1000


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

logger.info('Step 1 - Base data: loading MNIST from %s (auto-download if missing). '
            'MNIST provides the individual digits used to compose the ShuffleMNIST samples', DATA_DIR)
dataset_train =  torchvision.datasets.MNIST(DATA_DIR, train=True, download=True,
                             transform=torchvision.transforms.ToTensor())

dataset_test =  torchvision.datasets.MNIST(DATA_DIR, train=False, 
                                           download=True,transform=torchvision.transforms.ToTensor())
logger.info('MNIST loaded: %d train samples, %d test samples', len(dataset_train), len(dataset_test))

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train,drop_last=True, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size_test, shuffle = True,drop_last=True)
logger.info('MNIST DataLoaders created (batch_size_train=%d, batch_size_test=%d, shuffle=True, '
            'drop_last=True to keep batch shapes constant during composition)', batch_size_train, batch_size_test)


def train_func(loader):
    return Shuffledata.ShuffleMNIST(loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True,is_train=True)

def test_func(loader):
    return Shuffledata.ShuffleMNIST(loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True,is_train=False)
    
@timer
def new_func(loader, sh_func):
    shuffled_data = sh_func(loader)
    return shuffled_data

logger.info('Step 2 - Dataset synthesis: building ShuffleMNIST. Each sample pastes num=4 random digits '
            'at random non-overlapping anchors (min separation radius=42 px) on a wall_shape=112x112 canvas; '
            'sum=True makes the target the sum of the 4 digits')
shuffled_train = new_func(train_loader, train_func)
logger.info('Building ShuffleMNIST test set with identical parameters, from the held-out MNIST test split '
            'so no digit seen in training appears in validation')
shuffled_test = new_func(test_loader, test_func)

#shuffled_train = Shuffledata.ShuffleMNIST(train_loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True,is_train=True)
#shuffled_test = Shuffledata.ShuffleMNIST(test_loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True, is_train = False)


logger.info('There are %d images and %d labels in the train set.', len(shuffled_train.train_img),
        len(shuffled_train.train_label))
logger.info('There are %d images and %d labels in the test set.', len(shuffled_test.test_img),
        len(shuffled_test.test_label))

#Configuring shuffled DataLoader
from torch.utils.data.sampler import RandomSampler

#se cambian estos nombres a train loader para que sean los que se llaman en la red
train_sampler = RandomSampler(shuffled_train, replacement=True, num_samples= 51200, generator=None)
test_sampler = RandomSampler(shuffled_test, replacement=True, num_samples= 5760, generator=None)

trainshuffled_loader = torch.utils.data.DataLoader(shuffled_train, batch_size=batch_size_train
                                                   ,drop_last=False, sampler = train_sampler)

testshuffled_loader = torch.utils.data.DataLoader(shuffled_test, batch_size=batch_size_train
                                                  ,drop_last=False, sampler = test_sampler)
logger.info('Step 3 - Sampling: RandomSampler with replacement fixes the epoch size '
            '(51200 train / 5760 val samples per epoch) independently of the synthesized dataset size, '
            'giving uniform epochs and additional stochasticity')


#resnet18 = models.resnet18()
googlenet = models.googlenet()

#configurando la led para las targetas gráficas
#net = models.resnet18(pretrained=True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

logger.info('Step 4 - Hardware: using device %s (priority: CUDA > Apple MPS > CPU)', device)

logger.info('Step 5 - Model: GoogLeNet trained FROM SCRATCH (weights=None) to measure pure learning '
            'capacity on the synthetic task without ImageNet transfer; aux_logits=False since the '
            'auxiliary classifiers are an ImageNet-era training aid and are unsupported on MPS')
net = models.googlenet(weights=None, aux_logits=False, init_weights=True)
net = net.to(device)
net

criterion = nn.CrossEntropyLoss()

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 37)
net.fc = net.fc.to(device)
logger.info('Step 6 - Head replacement: final FC layer resized %d -> 37 outputs, one per possible '
            'digit sum (0..36 = 4 digits x max 9)', num_ftrs)

optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
logger.info('Step 7 - Optimization: SGD(lr=0.0001, momentum=0.9) chosen for stable convergence when '
            'training from scratch; CrossEntropyLoss as the standard multi-class classification objective')

#net = torch.nn.DataParallel(googlenet, device_ids=[0, 1, 2, 3])

#para el nombre de las imágenes
count_fig = 0
plot_filename = 'prueba6GoogleNetNotPretrained.png'

n_epochs = 500
print_every = 100
valid_loss_min = np.inf
patience = 10
epochs_no_improve = 0
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(trainshuffled_loader)
logger.info('Step 8 - Training loop: up to %d epochs, %d steps/epoch. Inputs are replicated from 1 to 3 '
            'channels (GoogLeNet expects RGB) and normalized with ImageNet statistics '
            '(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Early stopping with patience=%d epochs '
            'on validation loss prevents overfitting and wasted compute',
            n_epochs, total_step, patience)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    logger.info('Epoch %d started', epoch)
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
            logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f',
                   epoch, n_epochs, batch_idx, total_step, loss.item())
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    logger.info('Epoch %d training done - mean train-loss: %.4f, train-acc: %.4f%% (%d/%d correct)',
                epoch, np.mean(train_loss), 100 * correct/total, correct, total)
    batch_loss = 0
    total_t=0
    correct_t=0
    logger.info('Epoch %d validation started (net.eval + torch.no_grad: batchnorm frozen, no gradients, '
                'so the metric reflects pure generalization on unseen digit compositions)', epoch)
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
            data_t = data_t.to(device)


            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(testshuffled_loader))
        network_learned = batch_loss < valid_loss_min
        logger.info('Epoch %d validation done - mean val-loss: %.4f, val-acc: %.4f%% (%d/%d correct)',
                    epoch, np.mean(val_loss), 100 * correct_t/total_t, correct_t, total_t)

        #Queremos graficar el entrenamiento
        #Puede esto verse a 'tiempo real'??
        #mientras lo graficarmeos cada 100 epocs


        #count_fig =+ 1
        fig = plt.figure(figsize=(20,10))
        plt.title("Train-Validation Accuracy")
        plt.plot(train_acc, label='train')
        plt.plot(val_acc, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(plot_filename)
        logger.info('Train/validation accuracy curves updated in %s for visual inspection of '
                    'convergence and overfitting gap', plot_filename)

        
        if network_learned:
            epochs_no_improve = 0
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            logger.info('Improvement detected (val loss %.4f < previous best): checkpoint saved to resnet.pt. '
                        'Only the best-generalizing weights are persisted (model selection criterion)', batch_loss)
        else:
            epochs_no_improve += 1
            logger.info('No improvement for %d epoch(s), best val loss: %.4f', epochs_no_improve, valid_loss_min)
    net.train()

    if epochs_no_improve >= patience:
        logger.info('Early stopping triggered at epoch %d: validation loss did not improve for %d consecutive '
                    'epochs, indicating the network has stopped learning; best val loss achieved: %.4f',
                    epoch, patience, valid_loss_min)
        break

logger.info('Training finished after %d epoch(s). Best validation loss: %.4f. '
            'Best model stored in resnet.pt, accuracy curves in %s', epoch, valid_loss_min, plot_filename)