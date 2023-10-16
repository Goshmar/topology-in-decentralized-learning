import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class FullyConnected():
    def __init__(self, n):
        self.n = n

    def w(self, t=0, params=None):
        return torch.ones((self.n, self.n)) / self.n


class Solo():
    def __init__(self, n):
        self.n = n

    def w(self, t=0, params=None):
        return torch.eye(self.n)


class LinReg(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lin = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.lin(x)


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


X = fetch_california_housing(as_frame=True)['data']
y = fetch_california_housing(as_frame=True)['target']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Стандартизация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Конвертация массивов NumPy в тензоры PyTorch
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train.values)
y_test = torch.Tensor(y_test.values)

# Создание DataLoader для тренировочных данных
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

loss_fn = nn.MSELoss()
loss_test = nn.MSELoss()

# def train_for_W(Net, net_params, scheme, ds, num_steps, num_workers, batch_size, lr, loss_fn, thr, n_jobs=1):
#     workers = [Net(*net_params) for _ in range(num_workers)]
#     opts = [torch.optim.SGD(worker.parameters(), lr=lr) for worker in workers]
#
#     loss_for_W = []
#     for step in range(num_steps):
#         cur_losses = []
#
#         for idx in range(len(workers)):
#             worker = workers[idx]
#             batch = generate_batch(ds, batch_size)
#             X_batch = batch[0]
#             y_batch = batch[1]
#             outputs = worker(X_batch)
#             loss = loss_fn(outputs, y_batch.view(-1, 1))
#             worker.zero_grad()
#             loss.backward()
#             with torch.no_grad():
#                 for param in worker.parameters():
#                     param -= lr * param.grad
#
#             y_pred = worker(X_test)
#             mse = float(loss_fn(y_pred, y_test.view(-1, 1)))
#             cur_losses.append(mse)
#
#         cur_loss = np.mean(cur_losses)
#
#         if cur_loss < thr:
#             return step
#
#         print(cur_loss)
#
#         loss_for_W.append(cur_loss)
#
#         workers_params = [dict(worker.named_parameters()) for worker in workers]
#         for key in workers_params[0]:
#
#             true_shape = workers_params[0][key].shape
#             flat_shape = torch.prod(torch.tensor(workers_params[0][key].shape))
#             parameters_flattened = torch.empty(num_workers, flat_shape)
#             for i, param_dict in enumerate(workers_params):
#                 parameters_flattened[i] = torch.flatten(param_dict[key])
#
#             if hasattr(scheme, 'next_step'):
#                 W = scheme.get_w()
#                 scheme.next_step()
#             else:
#                 W = scheme.w(step)
#
#             new_parameters = torch.matmul(W, parameters_flattened)
#             for i, worker in enumerate(workers):
#                 worker.state_dict()[key].copy_(torch.reshape(new_parameters[i], true_shape))
#
#     return num_steps

loss_fn = nn.MSELoss()
loss_test = nn.MSELoss()

def train_for_W(Net, net_params, scheme, ds, num_steps, num_workers, batch_size, lr, loss_fn, thr, n_jobs=1):
    workers = [Net(*net_params) for _ in range(num_workers)]
    opts = [torch.optim.SGD(worker.parameters(), lr=lr) for worker in workers]

    loss_for_W = []
    Y0 = dict([])
    X0 = dict([])
    Y1 = dict([])
    X1 = dict([])
    previous_grads_flattened = dict([])

    for step in range(num_steps):
        cur_losses = []

        for idx in range(len(workers)):
            worker = workers[idx]
            batch = generate_batch(ds, batch_size)
            X_batch = batch[0]
            y_batch = batch[1]
            outputs = worker(X_batch)
            loss = loss_fn(outputs, y_batch.view(-1, 1))
            worker.zero_grad()
            loss.backward()
            #             with torch.no_grad():
            #                 for param in worker.parameters():
            #                     param -= lr * param.grad

            y_pred = worker(X_test)
            mse = float(loss_fn(y_pred, y_test.view(-1, 1)))
            cur_losses.append(mse)

        cur_loss = np.mean(cur_losses)
        print(cur_loss)

        workers_params = [dict(worker.named_parameters()) for worker in workers]

        for key in workers_params[0]:

            true_shape = workers_params[0][key].shape
            flat_shape = torch.prod(torch.tensor(workers_params[0][key].shape))
            parameters_flattened = torch.empty(num_workers, flat_shape)
            grads_flattened = torch.empty(num_workers, flat_shape)
            for i, param_dict in enumerate(workers_params):
                parameters_flattened[i] = torch.flatten(param_dict[key])
                grads_flattened[i] = torch.flatten(param_dict[key].grad)

            if hasattr(scheme, 'next_step'):
                W = scheme.get_w().to(device)
                scheme.next_step()
            else:
                W = scheme.w(step)

            if step == 0:
                Y0[key] = grads_flattened
                X0[key] = parameters_flattened
                X1[key] = W @ X0[key] - 0.05 * Y0[key]
                previous_grads_flattened[key] = grads_flattened
            else:
                X1[key] = W @ X0[key] - 0.05 * Y0[key]
                Y1[key] = W @ Y0[key] + (grads_flattened - previous_grads_flattened[key])
                X0[key] = X1[key]
                Y0[key] = Y1[key]
                previous_grads_flattened[key] = grads_flattened

            for i, worker in enumerate(workers):
                worker.state_dict()[key].copy_(torch.reshape(X1[key][i], true_shape))

    return num_steps


if __name__ == '__main__':
    train_for_W(
        Net=LinReg,
        net_params=[8],
        scheme=FullyConnected(10),
        ds=train_dataset,
        num_steps=10000,
        num_workers=10,
        batch_size=32,
        lr=0.001,
        loss_fn=nn.MSELoss(),
        thr=0,
        n_jobs=1
    )
