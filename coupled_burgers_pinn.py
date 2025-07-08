
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device and domain
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epsilon = 1.0
pi = np.pi
tmin, tmax = 0.0, 1.0
xmin, xmax = -pi, pi
lb = torch.tensor([tmin, xmin], dtype=torch.float32, device=device)
ub = torch.tensor([tmax, xmax], dtype=torch.float32, device=device)

def scale_input(x):
    return 2.0 * (x - lb) / (ub - lb) - 1.0

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.activation = Swish()

    def forward(self, x):
        return x + self.activation(self.linear(x))

class NetU(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 50)
        self.blocks = nn.Sequential(*[ResidualBlock(50) for _ in range(6)])
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)

class NetV(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 30)
        self.blocks = nn.Sequential(*[ResidualBlock(30) for _ in range(4)])
        self.output = nn.Linear(30, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)

class CoupledPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_u = NetU()
        self.net_v = NetV()

    def forward(self, x):
        x_scaled = scale_input(x)
        return self.net_u(x_scaled), self.net_v(x_scaled)

def residuals(model, X_r):
    X_r.requires_grad_(True)
    t, x = X_r[:, 0:1], X_r[:, 1:2]
    u, v = model(X_r)

    grads_u = torch.autograd.grad(u, X_r, torch.ones_like(u), create_graph=True)[0]
    grads_v = torch.autograd.grad(v, X_r, torch.ones_like(v), create_graph=True)[0]

    u_t, u_x = grads_u[:, 0:1], grads_u[:, 1:2]
    v_t, v_x = grads_v[:, 0:1], grads_v[:, 1:2]

    u_xx = torch.autograd.grad(u_x, X_r, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, X_r, torch.ones_like(v_x), create_graph=True)[0][:, 1:2]

    uv = u * v
    uv_x = torch.autograd.grad(uv, X_r, torch.ones_like(uv), create_graph=True)[0][:, 1:2]

    r1 = u_t - u_xx - 2 * u * u_x + uv_x
    r2 = v_t - v_xx - 2 * v * v_x + uv_x
    return r1, r2

def u_exact(x, t):
    return torch.exp(-t) * torch.sin(x)

def v_exact(x, t):
    return torch.exp(-t) * torch.sin(x)

def generate_data(N_0=100, N_b=100, N_r=5000):
    x_0 = (xmax - xmin) * torch.rand((N_0, 1), device=device) + xmin
    t_0 = torch.full_like(x_0, tmin)
    X_0 = torch.cat([t_0, x_0], dim=1)
    u_0 = u_exact(x_0, t_0)
    v_0 = v_exact(x_0, t_0)

    t_b = (tmax - tmin) * torch.rand((N_b, 1), device=device) + tmin
    x_b0 = torch.full_like(t_b, xmin)
    x_b1 = torch.full_like(t_b, xmax)
    X_b = torch.cat([torch.cat([t_b, x_b0], dim=1), torch.cat([t_b, x_b1], dim=1)], dim=0)
    u_b = u_exact(X_b[:, 1:2], X_b[:, 0:1])
    v_b = v_exact(X_b[:, 1:2], X_b[:, 0:1])

    t_r = (tmax - tmin) * torch.rand((N_r, 1), device=device) + tmin
    x_r = (xmax - xmin) * torch.rand((N_r, 1), device=device) + xmin
    X_r = torch.cat([t_r, x_r], dim=1)

    return X_0, u_0, v_0, X_b, u_b, v_b, X_r

def loss_fn(model, X_r, X_0, u_0, v_0, X_b, u_b, v_b,
            weight_pde=1.0, weight_ic=10.0, weight_bc=10.0):
    
    r1, r2 = residuals(model, X_r)
    loss_pde = torch.mean(r1**2) + torch.mean(r2**2)

    u_pred_0, v_pred_0 = model(X_0)
    loss_ic = torch.mean((u_pred_0 - u_0) ** 2) + torch.mean((v_pred_0 - v_0) ** 2)

    u_pred_b, v_pred_b = model(X_b)
    loss_bc = torch.mean((u_pred_b - u_b) ** 2) + torch.mean((v_pred_b - v_b) ** 2)

    total_loss = weight_pde * loss_pde + weight_ic * loss_ic + weight_bc * loss_bc
    return total_loss

def train_model(model, X_0, u_0, v_0, X_b, u_b, v_b, X_r,
                epochs=2000, patience=100, lr=1e-3):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, X_r, X_0, u_0, v_0, X_b, u_b, v_b)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.4e} | LR = {scheduler.get_last_lr()[0]:.2e}")

        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    print("Switching to LBFGS optimizer for fine-tuning...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=500, tolerance_grad=1e-8)

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = loss_fn(model, X_r, X_0, u_0, v_0, X_b, u_b, v_b)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    return model

def predict(model, N_plot=200):
    x_plot = torch.linspace(xmin, xmax, N_plot, device=device)
    t_plot = torch.full_like(x_plot, tmax)
    X_plot = torch.stack([t_plot, x_plot], dim=1)
    with torch.no_grad():
        u_pred, v_pred = model(X_plot)
    u_exact_vals = u_exact(x_plot, t_plot)
    v_exact_vals = v_exact(x_plot, t_plot)
    return x_plot.cpu().numpy(), u_pred.cpu().numpy(), v_pred.cpu().numpy(), u_exact_vals.cpu().numpy(), v_exact_vals.cpu().numpy()

def plot_results(x, u_pred, v_pred, u_exact, v_exact):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, u_exact, 'r', label='u_exact')
    plt.plot(x, u_pred, 'k--', label='u_pred')
    plt.title('u(x, t=1)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, v_exact, 'k--', label='v_exact')
    plt.plot(x, v_pred, 'b', label='v_pred')
    plt.title('v(x, t=1)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_error_metrics(model, t_eval=1.0, N_points=500):
    x = torch.linspace(xmin, xmax, N_points, device=device)
    t = torch.full_like(x, t_eval)
    X = torch.stack([t, x], dim=1)
    with torch.no_grad():
        u_pred, v_pred = model(X)
    u_true = u_exact(x, t)
    v_true = v_exact(x, t)

    u_err = torch.abs(u_pred.squeeze() - u_true)
    v_err = torch.abs(v_pred.squeeze() - v_true)

    u_Linf = u_err.max().item()
    u_L2 = torch.sqrt(torch.mean(u_err ** 2)).item()
    v_Linf = v_err.max().item()
    v_L2 = torch.sqrt(torch.mean(v_err ** 2)).item()

    print(f"\nError Metrics at t = {t_eval}")
    print(f"{'Metric':<12} {'u(x,t)':<12} {'v(x,t)':<12}")
    print(f"{'L_inf':<12} {u_Linf:<12.4e} {v_Linf:<12.4e}")
    print(f"{'L_2':<12} {u_L2:<12.4e} {v_L2:<12.4e}")

def save_plot(x, u_pred, v_pred, u_exact, v_exact, filename='pinn_plot_1.png'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, u_exact, 'k--', label='u_exact')
    plt.plot(x, u_pred, 'r', label='u_pred')
    plt.title('u(x, t=1)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, v_exact, 'k--', label='v_exact')
    plt.plot(x, v_pred, 'b', label='v_pred')
    plt.title('v(x, t=1)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")

# Run
model = CoupledPINN().to(device)
X_0, u_0, v_0, X_b, u_b, v_b, X_r = generate_data()
model = train_model(model, X_0, u_0, v_0, X_b, u_b, v_b, X_r)
x, u_pred, v_pred, u_ex, v_ex = predict(model)
plot_results(x, u_pred, v_pred, u_ex, v_ex)
compute_error_metrics(model)
save_plot(x, u_pred, v_pred, u_ex, v_ex)
