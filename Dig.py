import numpy as np
import torch


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


def train_for_W_DIG(Net, net_params, scheme, ds, num_steps, num_workers, batch_size, lr, loss_fn, thr, n_jobs=1):
    workers = [Net(*net_params) for _ in range(num_workers)]
    loss_for_step = []
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
            cur_losses.append(loss.item())

        cur_loss = np.mean(cur_losses)

        print(cur_loss)
        loss_for_step.append(cur_loss)
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
                W = scheme.get_w()
                scheme.next_step()
            else:
                W = scheme.w(step)

            if step == 0:
                Y0[key] = grads_flattened
                X0[key] = parameters_flattened
                X1[key] = W @ X0[key] - lr * Y0[key]
                previous_grads_flattened[key] = grads_flattened
            else:
                X1[key] = W @ X0[key] - lr * Y0[key]
                Y1[key] = W @ Y0[key] + (grads_flattened - previous_grads_flattened[key])
                X0[key] = X1[key]
                Y0[key] = Y1[key]
                previous_grads_flattened[key] = grads_flattened

            for i, worker in enumerate(workers):
                worker.state_dict()[key].copy_(torch.reshape(X1[key][i], true_shape))

    return np.min(loss_for_step)
