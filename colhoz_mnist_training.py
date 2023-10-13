import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from itertools import repeat
from tqdm import tqdm
from joblib import Parallel, delayed

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='.', train=False, transform=transform)

BATCH_SIZE = 1024
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE)


class DatasetForLR(Dataset):
    def __init__(self, num_samples, num_features, noise_scale=1):
        super().__init__()
        self.size = num_samples
        self.dataset = torch.randn((num_samples, num_features))
        self.weights = torch.randn(num_features)
        self.noise_scale = noise_scale

    def __getitem__(self, idx):
        return self.dataset[idx], self.dataset[idx].dot(self.weights) + torch.randn(1) * self.noise_scale

    def __len__(self):
        return self.size


class LinReg(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lin = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.lin(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.proj = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = torch.amax(x, dim=(2, 3))
        x = self.proj(x)
        return x


device = 'mps'

train_loader = repeater(iter(train_loader))
test_loader = iter(test_loader)


def process_batch(model, opt, batch, loss_fn):
    X_batch = batch[0].to(device)
    y_batch = batch[1].to(device)
    outputs = model(X_batch)

    loss = loss_fn(outputs, y_batch)
    loss.backward()
    opt.step()
    opt.zero_grad()

    return loss.item()


def generate_batch(ds, batch_size):
    ids = torch.randint(0, len(ds), (batch_size,))
    X = []
    y = []
    for idx in ids:
        X.append(ds[idx][0])
        if not torch.is_tensor(ds[idx][1]):
            y.append(torch.tensor(ds[idx][1]))
        else:
            y.append(ds[idx][1])
    return torch.stack(X, dim=0), torch.stack(y, dim=0)


def one_step_for_worker(workers, opts, idx, ds, batch_size, loss):
    worker = workers[idx]
    opt = opts[idx]
    batch = generate_batch(ds, batch_size)
    worker_loss = process_batch(worker, opt, batch, loss)
    return worker_loss


def train_for_W(Net, net_params, scheme, ds, num_steps, num_workers, batch_size, lr, loss, thr, n_jobs=1):
    workers = [Net(*net_params).to(device) for _ in range(num_workers)]
    opts = [torch.optim.SGD(worker.parameters(), lr=lr) for worker in workers]
    loss_for_W = []
    for step in range(num_steps):
        cur_loss = 0.0
        # for worker, opt in zip(workers, opts):

        cur_losses = Parallel(n_jobs=n_jobs)(delayed(one_step_for_worker)(workers,
                                                                          opts,
                                                                          idx,
                                                                          ds,
                                                                          batch_size,
                                                                          loss) for idx in range(len(workers)))

        cur_loss = np.mean(cur_losses)

        if cur_loss < thr:
            return step

        loss_for_W.append(cur_loss)
        # print(f'Step = {step}, loss = {cur_loss}')
        workers_params = [dict(worker.named_parameters()) for worker in workers]
        for key in workers_params[0]:

            true_shape = workers_params[0][key].shape
            flat_shape = torch.prod(torch.tensor(workers_params[0][key].shape))
            parameters_flattened = torch.empty(num_workers, flat_shape).to(device)
            for i, param_dict in enumerate(workers_params):
                parameters_flattened[i] = torch.flatten(param_dict[key])

            if hasattr(scheme, 'next_step'):
                W = scheme.get_w().to(device)
                scheme.next_step()
            else:
                W = scheme.w(step).to(device)

            new_parameters = torch.matmul(W, parameters_flattened)
            for i, worker in enumerate(workers):
                worker.state_dict()[key].copy_(torch.reshape(new_parameters[i], true_shape))

    return num_steps


W_fully = (torch.ones((100, 100)) / 100).to(device)
W_eye = torch.eye(100).to(device)

lin_reg_ds = DatasetForLR(256, 128)

from STAT_TEST import TimeVaringErdos, Erdos

scheme_var = TimeVaringErdos(n=64, p=1 / 8)
scheme_const = Erdos(n=64, p=1 / 8)


def loss(outputs, y_batch):
    return ((outputs - y_batch) ** 2).mean()


var_results = []
const_results = []
for lr in tqdm(np.linspace(0.01, 0.1, num=10)):
    var_results.append(train_for_W(LinReg,
                                   [128],
                                   scheme_var,
                                   lin_reg_ds,
                                   num_steps=100,
                                   num_workers=64,
                                   batch_size=1,
                                   lr=lr,
                                   loss=loss,
                                   thr=2,
                                   n_jobs=1))

    const_results.append(train_for_W(LinReg,
                                     [128],
                                     scheme_const,
                                     lin_reg_ds,
                                     num_steps=100,
                                     num_workers=64,
                                     batch_size=1,
                                     lr=lr,
                                     loss=loss,
                                     thr=2,
                                     n_jobs=1))
    print(var_results[-1], const_results[-1])
print(f'Best for var: {min(var_results)}')
print(f'Best for const: {min(const_results)}')
