import logging
import os
import ssl
from datetime import datetime

import certifi
# NTSNet/core/resnet.py downloads pretrained ResNet-50 weights via torch.hub, which
# uses the stdlib ssl module's default CA bundle. On some systems (e.g. python.org
# installs on macOS) that bundle is empty/stale and the download fails with
# CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundle instead, process-wide.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from NTSNet.config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, INPUT_SIZE
from NTSNet.core import model
from ShuffleMNIST import dataset as Shuffledata
from utils import timer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('ntsnet.log')],
)
logger = logging.getLogger('NTSNet')

logger.info('=' * 70)
logger.info('Experiment: NTS-Net (attention_net, ResNet-50 backbone) on ShuffleMNIST')
logger.info('Task: predict the SUM of 4 MNIST digits pasted on a %dx%d canvas',
            INPUT_SIZE[0], INPUT_SIZE[1])
logger.info('=' * 70)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
logger.info('Hardware: using device %s (priority: CUDA > Apple MPS > CPU)', device)

batch_size_train = 64
batch_size_test = 1000

NUM_CLASSES = 37  # sum of 4 MNIST digits (0-9 each) ranges 0..36

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

logger.info('Base data: loading MNIST from %s (auto-download if missing), '
            'shared with RESNet.py', DATA_DIR)
dataset_train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())
logger.info('MNIST loaded: %d train samples, %d test samples', len(dataset_train), len(dataset_test))

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, drop_last=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True, drop_last=True)


def train_func(loader):
    return Shuffledata.ShuffleMNIST(loader, anchors=[], num=4, radius=42,
                                    wall_shape=INPUT_SIZE[0], sum=True, is_train=True)


def test_func(loader):
    return Shuffledata.ShuffleMNIST(loader, anchors=[], num=4, radius=42,
                                    wall_shape=INPUT_SIZE[0], sum=True, is_train=False)


@timer
def new_func(loader, sh_func):
    return sh_func(loader)


logger.info('Dataset synthesis: wall_shape=%d matches the NTS-Net INPUT_SIZE so the '
            'anchor maps are generated for the actual image size (no resize needed)', INPUT_SIZE[0])
shuffled_train = new_func(train_loader, train_func)
shuffled_test = new_func(test_loader, test_func)

logger.info('There are %d images and %d labels in the train set.',
            len(shuffled_train.train_img), len(shuffled_train.train_label))
logger.info('There are %d images and %d labels in the test set.',
            len(shuffled_test.test_img), len(shuffled_test.test_label))

from torch.utils.data.sampler import RandomSampler

train_sampler = RandomSampler(shuffled_train, replacement=True, num_samples=51200, generator=None)
test_sampler = RandomSampler(shuffled_test, replacement=True, num_samples=5760, generator=None)

trainloader = torch.utils.data.DataLoader(shuffled_train, batch_size=BATCH_SIZE,
                                          drop_last=False, sampler=train_sampler)
testloader = torch.utils.data.DataLoader(shuffled_test, batch_size=BATCH_SIZE,
                                         drop_last=False, sampler=test_sampler)
logger.info('Sampling: RandomSampler with replacement (51200 train / 5760 test samples per epoch), '
            'training batch size %d from NTSNet.config', BATCH_SIZE)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prepare_batch(data_):
    # (B, H, W) single-channel walls -> (B, 3, H, W) normalized with ImageNet stats
    img = data_.unsqueeze(1).repeat(1, 3, 1, 1)
    return normalize(img)


# define model
start_epoch = 1
net = model.attention_net(topN=PROPOSAL_NUM)

# NTSNet/core/model.py hardcodes all three heads to CUB-200's 200 classes; resize them
# here for the 37-class ShuffleMNIST digit-sum task without touching the vendored file.
net.pretrained_model.fc = nn.Linear(net.pretrained_model.fc.in_features, NUM_CLASSES)
net.concat_net = nn.Linear(net.concat_net.in_features, NUM_CLASSES)
net.partcls_net = nn.Linear(net.partcls_net.in_features, NUM_CLASSES)

if resume:
    ckpt = torch.load(resume, map_location=device)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    logger.info('Resumed from %s at epoch %d', resume, start_epoch)
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
optimizers = [raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer]
schedulers = [MultiStepLR(opt, milestones=[60, 100], gamma=0.1) for opt in optimizers]

net = net.to(device)
net = DataParallel(net)

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'save_dir', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(save_dir, exist_ok=True)
logger.info('Checkpoints will be saved to %s every %d epoch(s)', save_dir, SAVE_FREQ)


def next_plot_filename(directory, base_name, ext='.png'):
    n = 1
    while os.path.exists(os.path.join(directory, f'{base_name}_{n}{ext}')):
        n += 1
    return os.path.join(directory, f'{base_name}_{n}{ext}')


plot_filename = next_plot_filename(os.path.dirname(os.path.abspath(__file__)), 'ntsnet_accuracy')
logger.info('Train/validation accuracy curve for this run will be saved to %s', plot_filename)


@timer
def entrenamiento(start_epoch, trainloader, testloader, net, creterion, optimizers, schedulers, device):
    n_epochs = 500
    patience = 10
    epochs_no_improve = 0
    test_loss_min = float('inf')
    eval_epochs = []
    train_acc_history = []
    test_acc_history = []
    for epoch in range(start_epoch, n_epochs):
        # begin training
        logger.info('Epoch %d started', epoch)
        net.train()
        for batch_idx, (data_, target_) in enumerate(trainloader):
            img = prepare_batch(data_).to(device)
            label = target_.to(device)
            batch_size = img.size(0)
            for optimizer in optimizers:
                optimizer.zero_grad()

            raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
            part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                        label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
            raw_loss = creterion(raw_logits, label)
            concat_loss = creterion(concat_logits, label)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

            total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            if batch_idx % 20 == 0:
                logger.info('Epoch %d, Step [%d/%d], total loss: %.4f (raw %.4f, rank %.4f, '
                            'concat %.4f, partcls %.4f)',
                            epoch, batch_idx, len(trainloader), total_loss.item(), raw_loss.item(),
                            rank_loss.item(), concat_loss.item(), partcls_loss.item())

        for scheduler in schedulers:
            scheduler.step()

        if epoch % SAVE_FREQ == 0:
            train_loss = 0
            train_correct = 0
            total = 0
            net.eval()
            for batch_idx, (data_t, target_t) in enumerate(trainloader):
                with torch.no_grad():
                    img = prepare_batch(data_t).to(device)
                    label = target_t.to(device)
                    batch_size = img.size(0)
                    _, concat_logits, _, _, _ = net(img)
                    # calculate loss
                    concat_loss = creterion(concat_logits, label)
                    # calculate accuracy
                    _, concat_predict = torch.max(concat_logits, 1)
                    total += batch_size
                    train_correct += torch.sum(concat_predict.data == label.data)
                    train_loss += concat_loss.item() * batch_size

            train_acc = float(train_correct) / total
            train_loss = train_loss / total
            logger.info('epoch:%d - train loss: %.3f and train acc: %.3f total sample: %d',
                        epoch, train_loss, train_acc, total)

            # evaluate on test set
            test_loss = 0
            test_correct = 0
            total = 0
            for batch_idx, (data_t, target_t) in enumerate(testloader):
                with torch.no_grad():
                    img = prepare_batch(data_t).to(device)
                    label = target_t.to(device)
                    batch_size = img.size(0)
                    _, concat_logits, _, _, _ = net(img)
                    # calculate loss
                    concat_loss = creterion(concat_logits, label)
                    # calculate accuracy
                    _, concat_predict = torch.max(concat_logits, 1)
                    total += batch_size
                    test_correct += torch.sum(concat_predict.data == label.data)
                    test_loss += concat_loss.item() * batch_size

            test_acc = float(test_correct) / total
            test_loss = test_loss / total
            logger.info('epoch:%d - test loss: %.3f and test acc: %.3f total sample: %d',
                        epoch, test_loss, test_acc, total)

            # early stopping: stop once test loss hasn't improved for `patience` evaluations
            if test_loss < test_loss_min:
                test_loss_min = test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logger.info('No improvement for %d evaluation(s), best test loss: %.4f',
                            epochs_no_improve, test_loss_min)

            # plot accuracy curve
            eval_epochs.append(epoch)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            fig = plt.figure(figsize=(20, 10))
            plt.title('NTS-Net Train-Validation Accuracy')
            plt.plot(eval_epochs, train_acc_history, label='train')
            plt.plot(eval_epochs, test_acc_history, label='validation')
            plt.xlabel('num_epochs', fontsize=12)
            plt.ylabel('accuracy', fontsize=12)
            plt.legend(loc='best')
            plt.savefig(plot_filename)
            plt.close(fig)
            logger.info('Accuracy curve updated in %s', plot_filename)

            # save model
            net_state_dict = net.module.state_dict()
            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

            if epochs_no_improve >= patience:
                logger.info('Early stopping triggered at epoch %d: test loss did not improve for %d '
                            'consecutive evaluation(s); best test loss: %.4f', epoch, patience, test_loss_min)
                break


entrenamiento(start_epoch, trainloader, testloader, net, creterion, optimizers, schedulers, device)

logger.info('finishing training')
