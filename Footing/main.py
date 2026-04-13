"""
@author: Pin ZHANG, 2026, Singapore
@Reference: A Comprehensive Investigation of Physics-Informed Learning in Forward and Inverse Analysis of Elastic and Elastoplastic Footing.
            Computers and Geotechnics, 181, 107110
"""
from tqdm import trange
import torch
from torch.optim import lr_scheduler
torch.cuda.empty_cache()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch.nn as nn
import warnings
from utility_integration import get_gauss_integration_points_2D, gauss_training_data_2D, get_gauss_integration_points_1D, gauss_training_data_1D

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

np.random.seed(2026)
torch.manual_seed(42)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation='tanh'):
        super(NeuralNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Softplus())

        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Softplus())

        layers.append(nn.Linear(hidden_size[-1], output_size))

        self.network = nn.Sequential(*layers)

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x_scale = x / 10.0
        return self.network(x_scale)




mse_loss = torch.nn.MSELoss()


def auto_grad(u, x):
    return torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]




class PINNs():
    def __init__(
        self,
        X_train, X_b,
        H, B, q, b, lmbd, mu,
        layers, activation='tanh', device=device,
        initial_lr=1.0, sadap = False
    ):

        self.device = device

        self.x_in = X_train[:, 0:1].requires_grad_(True).to(device).double()
        self.y_in = X_train[:, 1:2].requires_grad_(True).to(device).double()
        self.x_b = X_b[:, 0:1].requires_grad_(True).to(device).double()
        self.y_b = X_b[:, 1:2].requires_grad_(True).to(device).double()

        self.H = H
        self.B = B
        self.q = q
        self.b = b
        self.lmbd = lmbd
        self.mu = mu

        self.dnn = NeuralNetwork(
            input_size=layers[0], output_size=layers[-1], hidden_size=layers[1:-1],
            activation=activation,
        ).to(device).double()


        self.sadap = sadap


        # self.optimizer_Adam = torch.optim.Adam(params=self.dnn.parameters(), lr=initial_lr)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=initial_lr,
            max_iter=9,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1e-8,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        if self.sadap:
            self.scheduler = lr_scheduler.StepLR(self.optimizer_lbfgs, step_size=1000, gamma=0.1)

        self.iter = 0

        self.potential_loss = []
        self.int_energy_loss = []
        self.neuman_energy_loss = []




    def net_uv(self, x, y):
        uv = self.dnn(torch.cat([x, y], dim=1))
        u = x * y * (x - self.B * torch.ones_like(x)) * uv[:, 0:1]
        v = y * uv[:, 1:2]

        u_x = auto_grad(u, x)
        u_y = auto_grad(u, y)
        v_x = auto_grad(v, x)
        v_y = auto_grad(v, y)

        exx = u_x
        eyy = v_y
        exy = (u_y + v_x) / 2
        sxx = (self.lmbd+2*self.mu)*exx + self.mu*eyy
        syy = (self.lmbd+2*self.mu)*eyy + self.mu*exx
        szz = self.lmbd * (exx + eyy)
        sxy = 2*self.mu*exy

        return u, v, exx, eyy, exy, sxx, syy, szz, sxy

    def int_energy(self, x, y):
        _, _, exx, eyy, exy, sxx, syy, szz, sxy = self.net_uv(x, y)
        in_energy = torch.sum(0.5 * (sxx * exx + syy * eyy + sxy * exy)) * self.B * self.H / (4*elements_y*elements_x) #1/2hk
        return in_energy

    def ext_energy(self, x, y):
        v_b = self.net_uv(x, y)[1]
        neumann_energy = torch.sum(self.q * v_b) * self.b / (2*elements)
        return neumann_energy

    def energy_loss(self, td=None):
        in_energy = self.int_energy(self.x_in, self.y_in)
        neumann_energy = self.ext_energy(self.x_b, self.y_b)

        potential = in_energy - neumann_energy

        if td is not None:
            description = f"in: {in_energy.item():.6f}, out: {neumann_energy.item():.6f}, potential: {potential.item():.6f}, "
            description += f"iter: {self.iter}, lr: {self.optimizer_lbfgs.param_groups[0]['lr']:.6f}"
            td.set_description(description)

        return potential, in_energy, neumann_energy



    def train(self, epochs):
        self.dnn.train()

        with trange(epochs, dynamic_ncols=True, ncols=1) as td:
            for epoch in td:

                in_energy_tem = []
                neumann_energy_tem = []

                def closure():

                    self.optimizer_lbfgs.zero_grad()
                    loss, in_energy, neumann_energy = self.energy_loss(td=td)
                    in_energy_tem.append(in_energy)
                    neumann_energy_tem.append(neumann_energy)
                    #
                    # l2_reg = torch.tensor(0.).to(device)
                    # for param in self.dnn.parameters():
                    #     l2_reg += torch.norm(param, 2)
                    #
                    # loss += 0.002 * l2_reg
                    loss.backward()
                    return loss


                self.optimizer_lbfgs.step(closure)
                self.iter += 1

                self.potential_loss.append(closure().item())
                self.int_energy_loss.append(in_energy_tem[-1].item())
                self.neuman_energy_loss.append(neumann_energy_tem[-1].item())

                if self.sadap:
                    self.scheduler.step()

    def predict(self, x, y):

        self.dnn.eval().cpu()
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        u, v, exx, eyy, exy, sxx, syy, szz, sxy = self.net_uv(x, y)

        self.dnn.to(self.device)

        return u.detach(), v.detach(), \
               exx.detach(), eyy.detach(), exy.detach(), \
               sxx.detach(), syy.detach(), szz.detach(), sxy.detach(),


B = 10.0
H = 10.0
b = 2.0
q = 1500 #kN
# q = 15 #kN
E = 15000 #kpa
# E = 150 #kpa
nu = 0.3
mu = E/2/(1+nu)
lmbd = nu*E/(1+nu)/(1-2*nu)


elements_x = elements_y = 20
elements = 20

X_train = gauss_training_data_2D(width=B, height=H, elements_x=elements_x, elements_y=elements_y)

X_b = gauss_training_data_1D(integration_range=(0, b), elements=elements, y_corrd=H)

layers = [2, 64, 64, 64, 2]
pinn_model = PINNs(
    X_train=X_train, X_b=X_b,
    H=H, B=B, q=q, b=b, lmbd=lmbd, mu=mu,
    layers=layers, activation='tanh', device=device,
    sadap=True, initial_lr=1.0,
)

pinn_model.train(epochs=5000)

#####################################################################
x_plot = torch.linspace(0, 10, 101).double()
y_plot = torch.linspace(0, 10, 101).double()
X_plot, Y_plot = torch.meshgrid(x_plot, y_plot)
X_star = X_plot.flatten().unsqueeze(-1)
Y_star = Y_plot.flatten().unsqueeze(-1)


U_pred, V_pred, Exx_pred, Eyy_pred, Exy_pred, Sxx_pred, Syy_pred, Szz_pred, Sxy_pred = pinn_model.predict(X_star, Y_star)

d_prediction = pd.DataFrame({
                            'U_pred': U_pred.numpy().flatten(), 'V_pred': V_pred.numpy().flatten(),
                            'Exx_pred': Exx_pred.numpy().flatten(), 'Eyy_pred': Eyy_pred.numpy().flatten(),
                            'Exy_pred': Exy_pred.numpy().flatten(),
                            'Sxx_pred': Sxx_pred.numpy().flatten(), 'Syy_pred': Syy_pred.numpy().flatten(),
                            'Szz_pred': Szz_pred.numpy().flatten(),'Sxy_pred': Sxy_pred.numpy().flatten(),
                            })
