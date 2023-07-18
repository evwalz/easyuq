import torch
import time
import numpy as np


# FUNCTION:
def algo72_ensemble_torch(fct_train_np, fct_test_np, y_train_np):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fct_train = torch.from_numpy(fct_train_np).to(device)
    fct_test = torch.from_numpy(fct_test_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)

    n = fct_train.size(0)
    m = fct_test.size(0)
    sum_x = torch.sum(fct_train)
    sum_x2 = torch.sum(fct_train ** 2)

    XX = torch.zeros((2, 2)).to(device)
    XX[0, 0] = sum_x2
    XX[0, 1] = XX[1, 0] = -1 * sum_x
    XX[1, 1] = n
    XX /= (n * sum_x2 - (sum_x) ** 2)

    X_tr = torch.ones(n, 2).to(device)
    X_tr[:, 1] = fct_train
    X_tr_XX = X_tr @ XX
    H = X_tr_XX @ X_tr.t()
    C = torch.zeros(m, n).to(device)
    g2 = XX[0, 0] + XX[1, 0] * fct_test + fct_test * (XX[0, 1] + fct_test * XX[1, 1])
    g = ghelp = torch.zeros((n + 1)).to(device)
    Hbar = torch.zeros((n, n)).to(device)

    for j in range(m):
        g[0:n] = X_tr_XX[:, 0] + X_tr_XX[:, 1] * fct_test[j]
        g[n] = g2[j]
        Hbar = H - torch.outer(g[0:n], g[0:n]) / (1 - g[n])
        g /= (1 + g[n])
        gsrt = torch.sqrt(1 - g[n])
        Hdiagsqrt = torch.sqrt(1 - torch.diag(Hbar)[0:n])
        B = gsrt + g[0:n] / Hdiagsqrt
        A = torch.sum(g[0:n] * y_train) / gsrt + (y_train - torch.sum(Hbar[0:n, 0:n].t() * y_train, dim=1)) / Hdiagsqrt
        C[j, :] = A / B

    return C.cpu().numpy()  # convert the result back to numpy array


def algo72_ensemble_torch_tensor_on_gpu(fct_train, fct_test, y_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = fct_train.size(0)
    m = fct_test.size(0)
    sum_x = torch.sum(fct_train)
    sum_x2 = torch.sum(fct_train ** 2)

    XX = torch.zeros((2, 2), device=device)
    XX[0, 0] = sum_x2
    XX[0, 1] = XX[1, 0] = -1 * sum_x
    XX[1, 1] = n
    XX /= (n * sum_x2 - (sum_x) ** 2)

    X_tr = torch.ones(n, 2, device=device)
    X_tr[:, 1] = fct_train
    X_tr_XX = X_tr @ XX
    H = X_tr_XX @ X_tr.t()
    C = torch.zeros(m, n, device=device)
    g2 = XX[0, 0] + XX[1, 0] * fct_test + fct_test * (XX[0, 1] + fct_test * XX[1, 1])
    g = ghelp = torch.zeros((n + 1), device=device)
    Hbar = torch.zeros((n, n), device=device)

    for j in range(m):
        g[0:n] = X_tr_XX[:, 0] + X_tr_XX[:, 1] * fct_test[j]
        g[n] = g2[j]
        Hbar = H - torch.outer(g[0:n], g[0:n]) / (1 - g[n])
        g /= (1 + g[n])
        gsrt = torch.sqrt(1 - g[n])
        Hdiagsqrt = torch.sqrt(1 - torch.diag(Hbar)[0:n])
        B = gsrt + g[0:n] / Hdiagsqrt
        A = torch.sum(g[0:n] * y_train) / gsrt + (y_train - torch.sum(Hbar[0:n, 0:n].t() * y_train, dim=1)) / Hdiagsqrt
        C[j, :] = A / B

    return C


if __name__ == '__main__':
    np.random.seed(123)

    ######## Full loop with copying to and from GPU: ###################################
    
    fct_train_data_np = xr.open_dataset('./data/cnn_fct_train.nc')
    fct_test_data_np = xr.open_dataset('./data/cnn_fct_test.nc')
    y_data_np = xr.open_dataset('./data/cnn_obs_train.nc')

    dim_train = fct_train_data_np.t.shape[0]
    dim_test = fct_test_data_np.t.shape[0]
    grix_x = fct_train_data_np.t.shape[1]
    grid_y = fct_train_data_np.t.shape[2]

    t0 = time.time()
    dim_ens = y_data_np.shape[0]
    ens = np.zeros((grix_x, grid_y, dim_test, dim_train), dtype=np.float32)
    for i in range(grix_x):
        for j in range(grid_y):
            y_train = y_data_np[:, i, j]
            fct_train = fct_train_data_np[:, i, j]
            fct_test = fct_test_data_np[:, i, j]
            ens[i, j, :, :] = algo72_ensemble_torch(fct_train, fct_test, y_train)

    print(f'With copying to/fro GPU: Elapsed time = {time.time() - t0} seconds.')

    ######## Full loop without copying to and from GPU: ###################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    fct_train_data = torch.from_numpy(fct_train_data_np).to(device)
    fct_test_data = torch.from_numpy(fct_test_data_np).to(device)
    y_data = torch.from_numpy(y_data_np).to(device)

    dim_ens = y_data.shape[0]
    ens = torch.zeros((grix_x, grid_y, dim_test, dim_train), device=device)
    for i in range(grix_x):
        for j in range(grid_y):
            y_train = y_data[:, i, j]
            fct_train = fct_train_data[:, i, j]
            fct_test = fct_test_data[:, i, j]

            ens[i, j, :, :] = algo72_ensemble_torch_tensor_on_gpu(fct_train, fct_test, y_train)
    ens = ens.cpu().numpy()

    print(f'All on GPU: Elapsed time = {time.time() - t0} seconds.')
