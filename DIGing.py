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


def one_step_for_worker(workers, idx, ds, batch_size, loss_fn):
    worker = workers[idx]
    batch = generate_batch(ds, batch_size)
    X_batch = batch[0].to(device)
    y_batch = batch[1].to(device)
    outputs = worker(X_batch)
    loss = loss_fn(outputs, y_batch)
    loss.backward()

    return loss.item()


def train_for_W(Net, net_params, scheme, ds, num_steps, num_workers, batch_size, lr, loss, thr, n_jobs=1, optimizer='SGD', alpha=0.5):
    workers = [Net(*net_params).to(device) for _ in range(num_workers)]
    loss_for_W = []
    for step in range(num_steps):
        cur_loss = 0.0
        cur_losses = Parallel(n_jobs=n_jobs)(delayed(one_step_for_worker)(workers,
                                                                          idx,
                                                                          ds,
                                                                          batch_size,
                                                                          loss) for idx in range(len(workers)))
        cur_loss = np.mean(cur_losses)
        if cur_loss < thr:
            return step
        loss_for_W.append(cur_loss)

        workers_params = [dict(worker.named_parameters()) for worker in workers]
        Y0 = dict([])
        X0 = dict([])
        Y1 = dict([])
        X1 = dict([])
        previous_grads_flattened = dict([])
        if optimizer == 'SGD':
            for worker in workers:
                for param in worker.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad

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

        else:
            for key in workers_params[0]:
                true_shape = workers_params[0][key].shape
                flat_shape = torch.prod(torch.tensor(workers_params[0][key].shape))
                parameters_flattened = torch.empty(num_workers, flat_shape).to(device)
                grads_flattened = torch.empty(num_workers, flat_shape).to(device)
                for i, param_dict in enumerate(workers_params):
                    parameters_flattened[i] = torch.flatten(param_dict[key])
                    grads_flattened[i] = torch.flatten(param_dict[key].grads)

                if hasattr(scheme, 'next_step'):
                    W = scheme.get_w().to(device)
                    scheme.next_step()
                else:
                    W = scheme.w(step).to(device)

                if step == 0:
                    Y0[key] = grads_flattened
                    X0[key] = parameters_flattened
                    X1[key] = W @ X0[key] - alpha * Y0[key]
                else:
                    X1[key] = W @ X0[key] - alpha * Y0[key]
                    Y1[key] = W @ Y0[key] + (grads_flattened - previous_grads_flattened[key])

                for i, worker in enumerate(workers):
                    worker.state_dict()[key].copy_(torch.reshape(X1[key][i], true_shape))
                X0[key] = X1[key]
                previous_grads_flattened[key] = Y0[key]

    return num_steps
