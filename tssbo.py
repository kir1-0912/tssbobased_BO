import gpytorch
import torch
import torch.optim as optim
import math
from gpytorch.constraints.constraints import Interval
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
import numpy as np
import gpytorch.settings as gpts
from contextlib import ExitStack
from scipy.stats import norm
import math
import time
import pickle
import os

def quantize_tensor(x_tensor,bounds,precision=0.001):
    x_np=x_tensor.cpu().numpy()
    x_q_np=np.zeros_like(x_np)
    for i in range(x_np.shape[1]):
        low,high=bounds[i]
        step=precision
        x_q_np[:,i]=np.clip(
            np.round(x_np[:,i]/step)*step,
            low,high
        )
    return torch.tensor(x_q_np,dtype=torch.float64)




def sample_model_gradient(
        model_list,
        y_mean,
        y_std,
        X,
        delta=0.01,
):
    size_list = len(model_list)
    n_dim = X.size(-1)

    y_mean = y_mean.view(-1)
    y_std = y_std.view(-1)

    X_list = torch.tile(X.view(1, -1), torch.Size([n_dim, 1]))
    X_list = torch.cat([X_list + delta * torch.eye(n_dim), X_list - delta * torch.eye(n_dim)], 0)

    Y_sample = torch.empty(torch.Size([0, 2 * n_dim]))
    with torch.no_grad():
        for i in range(size_list):
            mvn = model_list[i](X_list)
            Y_sample = torch.cat(
                [Y_sample, model_list[i].likelihood(mvn).sample(torch.Size([1])) * (y_std[i] + 1e-6) + y_mean[i]], 0)
    if size_list > 1:
        Y_sample[1:, :] = Y_sample[1:, :].clip(min=0.0)

    Y_sample_plus = Y_sample[:, :n_dim].sum(0)
    Y_sample_minus = Y_sample[:, n_dim:].sum(0)
    g_sample = (Y_sample_plus - Y_sample_minus) / delta / 2

    return g_sample
class tSubspace:
    def __init__(self,
                 dim,
                 bounds,
                 W_prior=None,
                 mean=None,
                 start_f=None,
                 gradient_prior=None,
                 sigma=0.1,
                 mu=0.5,
                 c1=None,
                 c2=None,
                 allround_flag=False,
                 greedy_flag=False,
                 k=100
                 ):
        self.dim = dim
        if bounds is None:
            self.bounds = bounds
        else:
            self.bounds = torch.tensor(bounds, dtype=torch.float64)
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]
        if mean is None:
            self.mean = torch.rand(self.dim, dtype=torch.float64) * (self.ub - self.lb) + self.lb
        else:
            self.mean = torch.tensor(mean, dtype=torch.float64)

        self.mean = self.mean.view([self.dim, 1])

        if sigma is None:
            self.sigma = (torch.mean(self.ub) - torch.mean(self.lb)) / 5
        else:
            self.sigma = torch.tensor(sigma, dtype=torch.float64)

        self.mean_f = torch.tensor(start_f, dtype=torch.float64) if start_f is not None else None

        self.mu = mu
        if gradient_prior is None:
            self.prior = torch.zeros([dim, 1], dtype=torch.float64)
        else:
            self.prior = torch.tensor(gradient_prior, dtype=torch.float64).view([dim, 1])

        if c1 is None:
            c1 = (1 - math.exp(math.log(0.01) / k)) / 2

        if c2 is None:
            c2 = (1 - math.exp(math.log(0.01) / k)) / 2

        self._c_j = torch.tensor(c1, dtype=torch.float64)
        self._c_p = torch.tensor(c2, dtype=torch.float64)
        self._c_W = torch.tensor(1 - self._c_j - self._c_p, dtype=torch.float64)

        if W_prior is None:
            self._W = torch.eye(self.dim, dtype=torch.float64)
        else:
            self._W = torch.tensor(W_prior, dtype=torch.float64)

        self._chi_n = torch.tensor(math.sqrt(self.dim) * (
                1.0 - (1.0 / (4.0 * self.dim)) + 1.0 / (21.0 * (self.dim ** 2))
        ), dtype=torch.float64)

        self.allround_flag = allround_flag
        self.greedy_flag = greedy_flag
        self.value_sqrt = None
        self.Q = None

    def set_mean_f(self, mean_f):
        # Ensure mean_f is a tensor
        self.mean_f = torch.tensor(mean_f, dtype=torch.float64)

    def set_new_mean(self, mean, mean_f=None):
        self.mean = torch.tensor(mean, dtype=torch.float64)
        if mean_f is not None:
            self.mean_f = torch.tensor(mean_f, dtype=torch.float64)

    def _get_prior_gradient(self, prior_x, prior_y, alpha=0.01):
        assert self.mean_f is not None

        X_torch = torch.tensor(prior_x, dtype=torch.float64).view([self.dim, -1])
        Y_torch = torch.tensor(prior_y, dtype=torch.float64).view([-1, 1])

        Sk = X_torch - self.mean.view([self.dim, -1])

        Yk = Y_torch - self.mean_f

        J = torch.pinverse(Sk.mm(Sk.t()) + alpha * torch.eye(self.dim, dtype=torch.float64)).mm(Sk)
        J = J.mm(Yk)

        return J

    def desketch_gradient(self, X_torch, Y_torch):
        assert self.mean_f is not None

        X_torch = X_torch.double()
        Y_torch = Y_torch.double()

        Sk = X_torch - self.mean.view([self.dim, -1])

        Yk = Y_torch - self.mean_f

        Ak = Sk.mm(torch.linalg.pinv(Sk.t().mm(Sk)))

        Jk = self.prior + Ak.mm(Yk - Sk.t().mm(self.prior))

        return Jk.view([self.dim, -1])

    def normalize(self, x, bounds=None):
        if bounds is None:
            bounds = self.bounds
        lb = bounds[:, 0]
        ub = bounds[:, 1]

        if x.ndimension() == 2 and x.shape[1] == 1:
            return (x - lb.view(-1, 1)) / (ub.view(-1, 1) - lb.view(-1, 1))
        else:
            return (x - lb) / (ub - lb)

    def denormalize(self, x_norm, bounds=None):
        if bounds is None:
            bounds = self.bounds

        lb = bounds[:, 0]
        ub = bounds[:, 1]

        lb = lb.view(-1, 1)  # [dim, 1]
        ub = ub.view(-1, 1)  # [dim, 1]

        return x_norm * (ub - lb) + lb

    def compute_pk(self, X_torch, Y_torch):
        Y_arg = torch.argsort(Y_torch.ravel())
        X_torch = X_torch[:, Y_arg]

        if self.greedy_flag:
            weights = torch.zeros_like(Y_arg, dtype=torch.float64)
            weights[0] = 1
            self.mu_num = 1
            self.mu_eff = torch.tensor(1.0, dtype=torch.float64)
        else:
            weights_prime = torch.tensor(
                [
                    math.log((X_torch.size(1) + 1) * self.mu) - math.log(i + 1)
                    for i in range(X_torch.size(1))
                ],
                dtype=torch.float64
            )

            self.mu_num = math.floor(X_torch.size(1) * self.mu)
            self.mu_eff = (torch.sum(weights_prime[:self.mu_num]) ** 2) / torch.sum(weights_prime[:self.mu_num] ** 2)

            positive_sum = torch.sum(weights_prime[weights_prime > 0])
            negative_sum = torch.sum(torch.abs(weights_prime[weights_prime < 0]))
            weights = torch.where(
                weights_prime >= 0,
                1 / positive_sum * weights_prime,
                0.9 / negative_sum * weights_prime,
            )
        print(f"weight:{weights}")
        mean_standardized = self.normalize(self.mean)
        if self.allround_flag:
            X_delta = (X_torch - mean_standardized.view([self.dim, -1])) / self.sigma
            p_mu = torch.sum(X_delta * weights[:X_torch.size(1)].view([1, -1]), 1)
        else:
            X_mu = X_torch[:, :self.mu_num]
            X_mu_delta = (X_mu - mean_standardized.view([self.dim, -1])) / self.sigma
            p_mu = torch.sum(X_mu_delta * weights[:self.mu_num].view([1, -1]), 1)

        return p_mu.squeeze()

    def update_subspace(self, new_x, new_y, new_mean_f=None, GP_model_list=None, mean_and_std=None):
        X_torch = torch.as_tensor(new_x, dtype=torch.float64).view([self.dim, -1])
        Y_torch = torch.tensor(new_y, dtype=torch.float64).view([-1, 1])

        # X_mean = torch.mean(X_torch, dim=1, keepdim=True)
        # X_std = torch.std(X_torch, dim=1, keepdim=True)
        # X_standardized = (X_torch - X_mean) / X_std
        X_standardized = self.normalize(X_torch.T).T

        # Y_mean = torch.mean(Y_torch)
        # Y_std = torch.std(Y_torch)
        # Y_standardized = (Y_torch - Y_mean) / Y_std
        lb_y = torch.min(Y_torch)
        ub_y = torch.max(Y_torch)
        Y_standardized = (Y_torch - lb_y) / (ub_y - lb_y + 1e-8)

        if new_mean_f is not None:
            self.mean_f = torch.tensor(new_mean_f, dtype=torch.float64)

        Gk = None

        if GP_model_list is not None and mean_and_std is not None:
            Gk = sample_model_gradient(GP_model_list, mean_and_std[0], mean_and_std[1], self.mean, delta=0.01).view(
                [-1, 1])
            Gk = Gk / Gk.norm() * self._chi_n

        Jk = self.desketch_gradient(X_standardized, Y_standardized)

        print("Shape of X_standardized:", X_standardized.shape)
        print("Shape of Y_standardized:", Y_standardized.shape)
        Pk = self.compute_pk(X_standardized, Y_standardized)
        print("Shape of Pk before processing:", Pk.shape)
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        range_2_3 = (1 / 3) * (ub - lb)

        Pk_clamped = torch.clamp(Pk, min=-range_2_3, max=range_2_3)
        Pk = Pk_clamped


        print("\nIn update_subspace:")
        print(f"Pk: {Pk}")
        print(f"Pk norm: {Pk.norm()}")

        self.prior = Jk

        Jk = Jk / Jk.norm() * self._chi_n

        lb1 = [20, 20, 5, 5, 3, 5, 5, 3, 500, 500, 0.5, 0.5, 0.15]
        ub1 = [290, 290, 45, 17, 17, 45, 17, 17, 9500, 9500, 5.5, 5.5, 0.8]
        lb1 = torch.tensor(lb1, dtype=torch.float64)
        ub1 = torch.tensor(ub1, dtype=torch.float64)
        print(f"Before updating mean: {self.mean.ravel()}")
        pk11 = self.denormalize(Pk.view(-1, 1))
        pk1 = pk11.view(-1)
        self.mean = self.mean.ravel() + pk1.ravel() * self.sigma
        self.mean = torch.clamp(self.mean, min=lb1, max=ub1)
        print(f"After updating mean: {self.mean}")
        self.mean_f = None

        D, Q = self._eigen_decomposition()
        print("\nIn _eigen_decomposition:")
        print(f"Eigenvalues (D): {D}")
        print(f"Eigenvectors (Q): {Q}")

        W_2 = Q.mm(torch.diag(1 / D).mm(Q.t()))
        Pk = Pk.unsqueeze(1) if Pk.ndim == 1 else Pk
        Pk_normalized_norm = W_2.mm(Pk).norm()
        print('pk norm, ', Pk.norm())
        print('normalized pk norm, ', Pk_normalized_norm / self._chi_n - 1)
        print('another normalized pk norm, ', Pk_normalized_norm * torch.sqrt(self.mu_eff) / self._chi_n)

        c = torch.tensor((self.mu_eff + 2) / (self.mu_eff + self.dim + 5), dtype=torch.float64)
        self.sigma = self.sigma * torch.exp(c / (1 + c) * (Pk_normalized_norm / self._chi_n - 1))
        self.sigma = torch.clamp(self.sigma, min=0, max=0.2)
        print('sigma, ', self.sigma)

        if Gk is None:
            self._W = self._c_W * self._W + self._c_j * Jk.mm(Jk.t()) + self._c_p * Pk.mm(Pk.t())
        else:
            self._W = self._c_W * self._W + self._c_j * Jk.mm(Jk.t()) * 3 / 5 + self._c_p * Pk.mm(
                Pk.t()) * 4 / 5 + self._c_j * Gk.mm(Gk.t()) * 3 / 5
        self.value_sqrt = None
        self.Q = None

    def _eigen_decomposition(self):
        if self.value_sqrt is not None and self.Q is not None:
            return self.value_sqrt, self.Q

        W = self._W / 2 + self._W.t() / 2
        value, Q = torch.linalg.eigh(W)
        value_sqrt = torch.sqrt(torch.where(value > 1e-12, value, torch.tensor(1e-12, dtype=torch.float64)))

        self._W = Q.mm(torch.diag(value_sqrt ** 2)).mm(Q.t())
        self.value_sqrt = value_sqrt
        self.Q = Q

        return value_sqrt, Q

    # def sample_candidates(self, n_candidate=100, n_resample=10):
    #     D, B = self._eigen_decomposition()
    #     x = torch.empty(size=torch.Size([self.dim, 0]), dtype=torch.float64)
    #     for i in range(n_resample):
    #         if x.size(1) >= n_candidate:
    #             break
    #         z = torch.randn(self.dim, n_candidate, dtype=torch.float64)
    #         y = B.mm(torch.diag(D)).mm(z)
    #         x_candidate = self.mean.view([self.dim, 1]) + y * self.sigma
    #         if self.bounds is None:
    #             x = x_candidate
    #         else:
    #             inbox = torch.all(x_candidate > self.lb.view([-1, 1]), 0).multiply(
    #                 torch.all(x_candidate < self.ub.view([-1, 1]), 0)
    #             )
    #             if inbox.size(0):
    #                 x = torch.cat([x, x_candidate[:, inbox]], 1)
    #
    #     if x.size(1) < n_candidate:
    #         n_sample = n_candidate - x.size(1)
    #         z = torch.randn(self.dim, n_sample, dtype=torch.float64)
    #         y = B.mm(torch.diag(D)).mm(z)
    #         x_candidate = self.mean.view([self.dim, 1]) + y * self.sigma
    #         x = torch.cat([x, x_candidate], 1)
    #
    #     x = x.clip(min=self.lb.view([-1, 1]), max=self.ub.view([-1, 1]))
    #     x = x[:, :n_candidate]
    #     return x

    def sample_candidates(self, n_candidate=100, n_resample=10):
        D, B = self._eigen_decomposition()
        x = torch.empty(size=(self.dim, 0), dtype=torch.float64)

        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        margin = 1e-6



        for _ in range(n_resample):
            if x.size(1) >= n_candidate:
                break

            # z = torch.randn(self.dim, n_candidate, dtype=torch.float64)
            # y = B.mm(torch.diag(D)).mm(z)
            # print(f"Random noise (z): {z}")
            # print(f"Noise term (y): {y}")
            # mean_norm = self.normalize(self.mean).view(-1, 1)
            # mean_norm = mean_norm.expand(-1, n_candidate)
            # print(f"Normalized mean (mean_norm): {mean_norm}")
            # x_norm = mean_norm + y * self.sigma
            # x_norm = torch.clamp(x_norm, margin, 1.0 - margin)

            z = torch.randn(self.dim, n_candidate, dtype=torch.float64)
            z = z / torch.norm(z, dim=0, keepdim=True)
            print(f"Random noise (z): {z}")
            r = self.sigma
            y = z * r
            print(f"Noise term (y): {y}")
            mean_norm = self.normalize(self.mean).view(-1, 1)
            mean_norm = mean_norm.expand(-1, n_candidate)
            x_norm = mean_norm + y
            inbox_norm = torch.all((x_norm > 0.0) & (x_norm < 1.0), dim=0)
            if inbox_norm.any():
                x_norm_valid = x_norm[:, inbox_norm]
                x_candidate = self.denormalize(x_norm_valid)
                inbox_denorm = torch.all((x_candidate > lb.view([-1, 1])) & (x_candidate < ub.view([-1, 1])), dim=0)
                if inbox_denorm.any():
                    x = torch.cat([x, x_candidate[:, inbox_denorm]], dim=1)
            else:
                continue

            print(f"Normalized candidates (x_norm): {x_norm}")
            print(f"Candidates (x_candidate): {x_candidate}")
            # inbox = torch.all(x_candidate > lb.view([-1, 1]), dim=0) & \
            #         torch.all(x_candidate < ub.view([-1, 1]), dim=0)
            #
            # if inbox.any():
            #     x = torch.cat([x, x_candidate[:, inbox]], dim=1)

        if x.size(1) < n_candidate:
            n_sample = n_candidate - x.size(1)
            z = torch.randn(self.dim, n_sample, dtype=torch.float64)
            y = B.mm(torch.diag(D)).mm(z)
            mean_norm = self.normalize(self.mean).view(-1, 1)
            mean_norm = mean_norm.expand(-1, n_sample)
            x_norm = mean_norm + y * self.sigma
            inbox_norm = torch.all((x_norm > 0.0) & (x_norm < 1.0), dim=0)
            if inbox_norm.any():
                x_norm_valid = x_norm[:, inbox_norm]
                x_candidate = self.denormalize(x_norm_valid)
                inbox_denorm = torch.all((x_candidate > lb.view([-1, 1])) & (x_candidate < ub.view([-1, 1])), dim=0)
                if inbox_denorm.any():
                    x = torch.cat([x, x_candidate[:, inbox_denorm]], dim=1)
            # x_norm = torch.clamp(x_norm, margin, 1.0 - margin)
            # x_candidate = self.denormalize(x_norm)
            # x = torch.cat([x, x_candidate], dim=1)

        x = x.clip(min=lb.view([-1, 1]), max=ub.view([-1, 1]))
        return x[:, :n_candidate]


class NARGPKernel(gpytorch.kernels.Kernel):
    def __init__(self, n, kernel='RBF'):
        super(NARGPKernel, self).__init__()
        self.n = n
        if kernel == 'RBF':
            # ????????n-1???
            self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.n - 1, active_dims=torch.arange(self.n - 1))
            self.rbf2 = gpytorch.kernels.RBFKernel(ard_num_dims=self.n - 1, active_dims=torch.arange(self.n - 1))
            # ???????????
            self.rbf3 = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([self.n - 1]))
            self.mykernel = gpytorch.kernels.ScaleKernel(self.rbf2 * self.rbf3) + gpytorch.kernels.ScaleKernel(
                self.base_kernel)
        elif kernel == 'MAT52':
            self.base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.n - 1,
                                                             active_dims=torch.arange(self.n - 1))
            self.rbf2 = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.n - 1,
                                                      active_dims=torch.arange(self.n - 1))
            self.rbf3 = gpytorch.kernels.MaternKernel(nu=2.5, active_dims=torch.tensor([self.n - 1]))
            self.mykernel = gpytorch.kernels.ScaleKernel(self.rbf2 * self.rbf3) + gpytorch.kernels.ScaleKernel(
                self.base_kernel)

    # ????????????????Additive
    # Kernel??????????
    #
    # gpytorch.kernels.ScaleKernel(self.rbf2 * self.rbf3)
    # gpytorch.kernels.ScaleKernel(self.base_kernel)
    # ???????????????
    #
    # ???????
    # base_kernel
    # ????
    # n - 1
    # ????????????????????????
    # rbf2 * rbf3
    # ???????????????????????
    # ????????
    # ?????????????????????
    # ??????????????????????????
    # ???????
    # ?????????????????????????????????????
    # ????????????????????????????????
    # ???????
    # ??????????????????????????????
    # ???????????
    # ????????????????????????????????
    # ????????
    # ??????????????????????????????????????????
    # ???????????????????????
    #
    # ???????????????
    # ??????????????????????????
    # ?????????????????????????????

    def forward(self, x1, x2, **params):
        return self.mykernel(x1, x2)

class GP(ExactGP, GPyTorchModel):
    def __init__(self, dataset, kernel='RBF', inner_kernel='RBF', k=1):
        self.x_train = dataset['train_x']
        self.y_train = dataset['train_y'].view([-1])
        self.n_train = self.x_train.size(0)
        self.n_dim = self.x_train.size(1)
        # self.num_outputs = 1
        self.sigma_train = dataset['train_sigma']
        if self.sigma_train is not None:
            super(GP, self).__init__(self.x_train, self.y_train,
                                     gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self.sigma_train))
        else:
            super(GP, self).__init__(self.x_train, self.y_train,
                                     gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2)))
        # self.num_outputs = 1
        if kernel == 'RBF':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.n_dim)
        elif kernel == 'MAT52':
            base_kernel = gpytorch.kernels.MaternKernel(lengthscale_constraint=Interval(0.005, math.sqrt(self.n_dim)),
                                                        nu=2.5, ard_num_dims=self.n_dim)

        self.mean_module = gpytorch.means.ConstantMean()
        if k == 1:
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_constraint=Interval(0.05, 20))
        else:
            self.covar_module = NARGPKernel(self.n_dim, kernel=inner_kernel)

    def fit(self):
        self.train()
        self.likelihood.train()

        optimizer = optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        num_epochs = 100
        for _ in range(num_epochs):
            optimizer.zero_grad()
            output = self(self.x_train)
            loss = -mll(output, self.y_train)

            loss.backward()

            optimizer.step()

        self.eval()
        self.likelihood.eval()

    def forward(self, x):

        mean_x = self.mean_module(x)
        # print(mean_x.detach())

        covar_x = self.covar_module(x)
        # print(covar_x.evaluate())
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def predict(self, x, full_cov=0):
        # with torch.no_grad():
        pred = self(x)
        if full_cov:
            return pred.mean.detach(), pred.covariance_matrix.detach()
        else:
            return pred.mean.detach(), pred.variance.detach()

def select_training_set(
        X_center,
        X,
        Y,
        B=None,
        D=None,
        n_training=100,
        use_C=True,
):
    X_center = X_center.view([1, -1])
    X_diff = X - X_center

    if use_C:
        # print(B)
        # print(D)
        C_2 = B.mm(torch.diag(1 / D)).mm(B.t())
        X_diff = X_diff.mm(C_2.t())

    X_diff_norm = torch.linalg.norm(X_diff, axis=1)
    sortargs = torch.argsort(X_diff_norm)

    return X[sortargs[:n_training], ...], Y[sortargs[:n_training], ...]

def select_candidate_EI_unconstrained(
        X,
        Y,
        X_candidate,
        batch_size,
        # n_candidates,
        sampler='cholesky',
        use_keops=False,
        device='cpu',
        noise_flag=True,
        xi=0.05
):
    # X_torch = torch.tensor(X).to(device=device)
    # X_candidate_torch = torch.tensor(X_candidate).to(device=device)
    # Y_torch = torch.tensor(Y).to(device=device)
    dtype = torch.float64
    # X_torch = X.clone().detach().to(device=device)
    # X_candidate_torch = X_candidate.clone().detach().to(device=device)
    # Y_torch = Y.clone().detach().to(device=device)
    X_torch = X.clone().detach().to(device=device, dtype=dtype)
    X_candidate_torch = X_candidate.clone().detach().to(device=device, dtype=dtype)
    Y_torch = torch.tensor(Y, dtype=dtype).clone().detach().to(device=device)


    assert torch.all(torch.isfinite(Y_torch))
    y_mean, y_std = Y_torch.mean(), Y_torch.std()

    train_y = (Y_torch - Y_torch.mean()) / (1e-6 + Y_torch.std())
    model = GP({'train_x': X_torch, 'train_y': train_y, 'train_sigma': None}, kernel='MAT52')
    model.fit()

    mvn = model(X_candidate_torch)

    mean = mvn.mean.detach().numpy()
    var = mvn.variance.detach().numpy()
    best_f = Y_torch.min().detach().numpy()

    stdd = np.sqrt(var + 1e-12)
    normed = - (mean - best_f) / stdd
    EI = stdd * (normed * norm.cdf(normed) + norm.pdf(normed))
    log_EI = np.log(np.maximum(0.000001, EI))
    tmp = np.minimum(-40, normed) ** 2
    log_EI_approx = np.log(stdd) - tmp / 2 - np.log(tmp - 1)
    log_EI_approx = log_EI * (normed > -40) + log_EI_approx * (normed <= -40)

    ei_argsort = np.argsort(-1 * log_EI_approx)
    return X_candidate_torch[ei_argsort[:batch_size], :], [model], [y_mean, y_std]

def select_candidate_TS_unconstrained(
        X,
        Y,
        X_candidate,
        batch_size,
        # n_candidates,
        sampler='lanczos',
        use_keops=False,
        device='cpu',
        noise_flag=True,
):
    # X_torch = torch.tensor(X).to(device=device)
    # X_candidate_torch = torch.tensor(X_candidate).to(device=device)
    # Y_torch = torch.tensor(Y).to(device=device)
    dtype = torch.float64
    # X_torch = X.clone().detach().to(device=device)
    # X_candidate_torch = X_candidate.clone().detach().to(device=device)
    # Y_torch = Y.clone().detach().to(device=device)
    X_torch = X.clone().detach().to(device=device, dtype=dtype)
    X_candidate_torch = X_candidate.clone().detach().to(device=device, dtype=dtype)
    Y_torch = torch.tensor(Y, dtype=dtype).clone().detach().to(device=device)

    assert torch.all(torch.isfinite(Y_torch))
    y_mean, y_std = Y_torch.mean(), Y_torch.std()

    train_y = (Y_torch - Y_torch.mean()) / (1e-5 + Y_torch.std())

    model = GP({'train_x': X_torch, 'train_y': train_y, 'train_sigma': None}, kernel='MAT52')
    model.fit()

    with ExitStack() as es:
        if sampler == "cholesky":
            es.enter_context(gpts.max_cholesky_size(float("inf")))
        elif sampler == "ciq":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(
                gpts.minres_tolerance(2e-3)
            )  # Controls accuracy and runtime
            es.enter_context(gpts.num_contour_quadrature(15))
        elif sampler == "lanczos":
            es.enter_context(
                gpts.fast_computations(
                    covar_root_decomposition=True, log_prob=True, solves=True
                )
            )
            es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(False))
        elif sampler == "rff":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

    with torch.no_grad():
        y_cand_dist = model(X_candidate_torch)
        y_cand = model.likelihood(y_cand_dist).sample(torch.Size([batch_size])).t().cpu().detach()

    X_cand_select = torch.ones((batch_size, X_candidate_torch.size(-1)))
    for i in range(batch_size):
        # Pick the best point and make sure we never pick it again
        indbest = torch.argmin(y_cand[:, i])
        X_cand_select[i, :] = X_candidate_torch[indbest, :]
        y_cand[indbest, :] = torch.inf

    return X_cand_select, [model], [y_mean, y_std]

def tssbo_solver(funct, dim, bounds, init_x=None, init_y=None,
                 sigma=0.1, mu=0.5, c1=None, c2=None, allround_flag=False, greedy_flag=False,
                 n_training=None, batch_size=20, n_candidates=200, n_resample=10, nMax=100, k=100,
                 dataset_file='./dataset_tTs_bo.pkl', use_BO=True, use_TS=True,
                 calculate_model_gradient=True):
    # Part 1: Wrapper for the target function to ensure scalar output
    def t_funct(x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = np.array(x)

        # Ensure x_np is 2D
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        result = funct(x_np)  # Assume funct returns a scalar value now
        return float(result)  # Convert result to a Python scalar

    t00 = time.time()
    if n_training is None:
        n_training = min(dim * 2, 500)  # Default training size

    subspace = tSubspace(dim, bounds, sigma=sigma, mu=mu, c1=c1, c2=c2, allround_flag=allround_flag,
                         greedy_flag=greedy_flag, k=k)  # Initialize subspace
    model_list = None
    m_n_s = None
    evaluation_hist = ([], [])  # Store history as list of tuples (x, y)

    # Part 2: Initialization phase
    if init_x is not None:
        x = torch.tensor(init_x)
        if init_y is None:
            init_y = t_funct(x)

        # Update evaluation history with tuple (x, y)
        evaluation_hist[0].append(x)
        evaluation_hist[1].append(init_y)

        # Subspace initialization logic
        mean_prior = x.clone()  # Simplified since y is scalar
        f_mean_prior = init_y  # Directly store scalar y
        subspace.set_new_mean(mean_prior, f_mean_prior)

    # Part 3: Main optimization loop
    t1 = time.time()
    print('Initialization time:', (t1 - t00))
    g = 0
    while len(evaluation_hist[1]) <= nMax:
        g += 1
        print('Iteration:', g)

        t0 = time.time()
        print('Elapsed time:', t0 - t00)

        # Candidate generation
        X_candidates = subspace.sample_candidates(n_candidates, n_resample).t()
        # X_candidates = quantize_tensor(X_candidates, bounds, precision=0.001)

        t1 = time.time()
        print('Candidate generation time:', (t1 - t0))

        # Candidate selection using BO or random sampling
        if len(evaluation_hist[1]) and use_BO:
            X = torch.stack(evaluation_hist[0])
            Y = torch.tensor(evaluation_hist[1])  # Convert list of scalars to tensor
            X_center = subspace.mean.ravel()

            if X.size(0) >= n_training:
                D, B = subspace._eigen_decomposition()
                X, Y = select_training_set(
                    X_center,
                    X,
                    Y,
                    B,
                    D,
                    n_training=n_training
                )

            if use_TS:
                X_cand, model_list, m_n_s = select_candidate_TS_unconstrained(
                    X,
                    Y,
                    X_candidates,
                    batch_size=batch_size
                )
            else:
                X_cand, model_list, m_n_s = select_candidate_EI_unconstrained(
                    X,
                    Y,
                    X_candidates,
                    batch_size=batch_size
                )
        else:
            X_cand = X_candidates[:batch_size, :]

        t2 = time.time()
        print('Candidate selection time:', (t2 - t1))

        # Evaluate candidates
        if not calculate_model_gradient:
            model_list = None
            m_n_s = None

        if subspace.mean_f is None:
            X_cand = torch.cat([subspace.mean.view([1, -1]), X_cand], 0)
            Y_cand = [t_funct(x_i.numpy()) for x_i in X_cand]  # List comprehension for scalar outputs
            t3 = time.time()
            print(Y_cand)
            subspace.set_mean_f(Y_cand[0])  # Set first candidate's value as mean_f
            subspace.update_subspace(X_cand[1:, :].t(), Y_cand[1:], GP_model_list=model_list, mean_and_std=m_n_s)
            t4 = time.time()
        else:
            Y_cand = [t_funct(x_i.numpy()) for x_i in X_cand]
            t3 = time.time()
            subspace.update_subspace(X_cand.t(), Y_cand, GP_model_list=model_list, mean_and_std=m_n_s)
            t4 = time.time()

        print('Simulation time:', (t3 - t2))
        print('Update subspace time:', (t4 - t3))

        # Update evaluation history
        for x_cand, y_cand in zip(X_cand, Y_cand):
            evaluation_hist[0].append(x_cand)
            evaluation_hist[1].append(y_cand)

        print('Best y:', min(evaluation_hist[1]))

        # Save progress to file
        with open(dataset_file, 'wb') as f:
            pickle.dump({'x': evaluation_hist[0], 'y': evaluation_hist[1]}, f)
