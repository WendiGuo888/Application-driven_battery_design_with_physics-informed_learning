# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:45:52 2026

Figure 4 replica

@author: wengu476
"""
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import copy
import random
import torch
import pickle
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MultipleLocator, MaxNLocator
from scipy.ndimage import gaussian_filter1d

#%% Adaptive aging parameter update

def plot_physics_parameter_evolution_1x4_final(param_history):

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(
        1, 4,
        figsize=(13.5, 2.6),
        dpi=300
    )

    steps = np.arange(len(param_history["A"]))

    plot_cfg = [
        ("A", r"$A_u$"),
        ("B", r"$B_u$"),
        ("Z", r"$Z_u$"),
        ("Phi", r"$\Phi$")
    ]

    color_path = "#2F5D8C"
    ref_color  = "#C44E52"

    for i, (ax, (key, label)) in enumerate(zip(axes, plot_cfg)):
        data = np.asarray(param_history[key], dtype=float)


        ax.axhline(
            0,
            color=ref_color,
            linestyle="--",
            alpha=0.35,
            lw=1.0
        )


        ax.plot(
            steps,
            data,
            color=color_path,
            lw=2.0
        )

        ax.set_title(label, pad=4, fontweight="normal")


        ax.set_xlabel("Recording points", labelpad=5)

        if i == 0:
            ax.set_ylabel("Change [%]", labelpad=6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.tick_params(
            direction="out",
            length=3.5,
            width=1.0
        )
        ax.grid(False)


    plt.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.28,
        top=0.88,
        wspace=0.30
    )

    plt.show()

    mpl.rcParams.update(mpl.rcParamsDefault)
    
#%% Update the physics loss plot in paper style

def plot_physics_loss_1x3_compact(loss_history):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl

    FONT = 15
    mpl.rcParams.update({
        "font.size": FONT,
        "axes.labelsize": FONT + 1,
        "axes.titlesize": FONT + 2,
        "xtick.labelsize": FONT,
        "ytick.labelsize": FONT,
        "legend.fontsize": FONT - 1
    })


    phys_val = loss_history["l_phys_val"]
    d1 = loss_history["l_d1"]
    phys_total = loss_history["phys_total"]
    mono = loss_history["l_mono"]
    zero = loss_history["l_zero"]

    epochs = np.arange(len(phys_val))


    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, 3.2),  
        dpi=300
    )

    # ==========================================
    # 1️⃣ Physics Fidelity
    # ==========================================
    ax = axes[0]

    ax.plot(epochs, phys_val, color="#D62728", lw=2.2, label="Physics Value")
    ax.plot(epochs, d1, color="#2CA02C", lw=2.2, label="First Derivative")

    ax.set_title("Physics Fidelity", pad=6)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    # ==========================================
    # 2️⃣ Weighted Physics
    # ==========================================
    ax = axes[1]

    ax.plot(epochs, phys_total, color="#8A2BE2", lw=2.5,
            label="Sum of Physics Penalty")

    ax.set_title("Weighted Physics", pad=6)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    # ==========================================
    # 3️⃣ Physical Constraints
    # ==========================================
    ax = axes[2]

    ax.plot(epochs, mono, color="#F39C12", lw=2.2, label="Monotonicity")
    ax.plot(epochs, zero, color="#444444", lw=2.2, label="Zero Anchor")

    ax.set_title("Physical Constraints", pad=6)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")


    for ax in axes:
        ax.set_yscale("log") 

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.grid(False)

        ax.tick_params(direction="out", width=1.1, length=4)


        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.85,
            edgecolor="none"
        )

    plt.tight_layout(w_pad=2.0)
    plt.show()

    mpl.rcParams.update(mpl.rcParamsDefault)

#%%

def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

#%%

def prepare_cell_dataframe_transfer(
    df: pd.DataFrame,
    N_total: int = 16,
    use_phys: bool = True,   
    efc_col: str = "EFC",
    soh_col: str = "SOH",
    fixed_features: list = None,
    drop_cols: tuple = ("RUL",),
):
    df = df.copy()

    if soh_col in df.columns:
        df["Qloss"] = 1.0 - df[soh_col].astype(float)
    elif "Qloss" not in df.columns:
        raise KeyError(f"Neither '{soh_col}' nor 'Qloss' found.")

    if use_phys:
        phys_features = [c for c in df.columns if "_predicted" in c]
    else:
        phys_features = [] 
    
    n_phys = len(phys_features)
    

    if fixed_features is not None:
        top_ordered = fixed_features
    else:

        n_exp_needed = N_total - 1 - n_phys 

        exclude = {soh_col, "Qloss", "SOH", efc_col}
        for c in drop_cols:
            if c in df.columns: exclude.add(c)
        for c in phys_features:
            exclude.add(c)

        candidate_exp_features = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        corr = df[candidate_exp_features].corrwith(df["Qloss"], method="spearman").abs()
        corr = corr.dropna()
        top_exp = list(corr.sort_values(ascending=False).head(n_exp_needed).index)

        top_ordered = [efc_col] + phys_features + top_exp

    available_cols = [c for c in top_ordered if c in df.columns]
    out = df[available_cols].copy()
    out["Qloss"] = df["Qloss"].values
    
    return out, top_ordered

#%% cell cross transfer stategy

def split_data_for_transfer(
    data: pd.DataFrame,
    ratio_early: float = 0.3,     
    ratio_val: float = 0.2,      
    is_source: bool = False,      
    efc_col_name: str = "EFC",
):
 
    data = data.copy().sort_values(efc_col_name).reset_index(drop=True)
    N = len(data)

    if is_source:

        cut_split = int((1 - ratio_val) * N)
        data_train = data.iloc[:cut_split].reset_index(drop=True)
        data_val   = data.iloc[cut_split:].reset_index(drop=True)
        data_test  = pd.DataFrame() 
    else:

        cut_early = int(ratio_early * N)
        data_early = data.iloc[:cut_early]
        data_test  = data.iloc[cut_early:].reset_index(drop=True)


        N_early = len(data_early)
        cut_val = int((1 - ratio_val) * N_early)
        data_train = data_early.iloc[:cut_val].reset_index(drop=True)
        data_val   = data_early.iloc[cut_val:].reset_index(drop=True)

    return data_train, data_val, data_test

#%% source domain fit and target domain transfer

def scale_transfer_flow(
    data_src_tr, data_src_val,  
    data_tgt_tr, data_tgt_val,  
    data_tgt_te,               
    efc_col_name="EFC",
    target_col="Qloss",
    M_fixed=1200,
    clip_range=(-0.2, 1.2)
):

    def get_f_y_x(df):
        f = df[efc_col_name].to_numpy()
        y = df[target_col].to_numpy()
        x = df.drop(columns=[target_col]).copy()
        return f, y, x

    f_src_tr, y_src_tr, x_src_tr = get_f_y_x(data_src_tr)
    f_src_val, y_src_val, x_src_val = get_f_y_x(data_src_val)
    f_tgt_tr, y_tgt_tr, x_tgt_tr = get_f_y_x(data_tgt_tr)
    f_tgt_val, y_tgt_val, x_tgt_val = get_f_y_x(data_tgt_val)
    f_tgt_te, y_tgt_te, x_tgt_te = get_f_y_x(data_tgt_te)


    M = float(M_fixed) if M_fixed else float(np.max(f_src_tr))


    for df in [x_src_tr, x_src_val, x_tgt_tr, x_tgt_val, x_tgt_te]:
        df[efc_col_name] = df[efc_col_name] / M

        cols = [efc_col_name] + [c for c in df.columns if c != efc_col_name]
        df[:] = df[cols]


    scaler = MinMaxScaler()

    scaler.fit(x_src_tr.to_numpy()[:, 1:]) 

    def apply_scaling(df_x, y_np):
        x_np = df_x.to_numpy(dtype=float)
        x_rest = scaler.transform(x_np[:, 1:])
        x_rest = np.clip(x_rest, clip_range[0], clip_range[1])
        x_final = np.concatenate([x_np[:, [0]], x_rest], axis=1)
        
        return torch.tensor(x_final, dtype=torch.float32), \
               torch.tensor(y_np, dtype=torch.float32).view(-1, 1)


    X_src_tr, Q_src_tr = apply_scaling(x_src_tr, y_src_tr)
    X_src_val, Q_src_val = apply_scaling(x_src_val, y_src_val)
    X_tgt_tr, Q_tgt_tr = apply_scaling(x_tgt_tr, y_tgt_tr)
    X_tgt_val, Q_tgt_val = apply_scaling(x_tgt_val, y_tgt_val)
    X_tgt_te, Q_tgt_te = apply_scaling(x_tgt_te, y_tgt_te)

    return (X_src_tr, Q_src_tr, X_src_val, Q_src_val), \
           (X_tgt_tr, Q_tgt_tr, X_tgt_val, Q_tgt_val, X_tgt_te, Q_tgt_te), \
           M, scaler
           
#%%

from scipy.optimize import curve_fit

def physics(A, B, Phi, Z, M, F) :
    return (A * (Phi / Z) * ((1 + Z * F * M + Z) ** (1 / B) - (1 + Z) ** (1 / B)))

def fit_physics_params(F_train, Q_train, M_ref, trace = False):
    #
    M = M_ref  
    F = F_train / M

    # 

    initial = [1e-4, 0.26, 0.28, 0.001] 
    bounds = ([0, 0.1, 0.01, 0.001], [1e-3, 1.0, 1.0, 0.1])

    # 
    params, params_covariance = curve_fit(
        lambda F, A, B, Phi, Z: physics(A, B, Phi, Z, M, F),
        F, Q_train.ravel(),
        p0 = initial,
        bounds = bounds
    )
    
    #
    if (trace) : 
        print(f"Optimal parameters: \n\tA = {params[0]}, \n\tB = {params[1]}, \n\tPhi = {params[2]}, \n\tZ = {params[3]}")
    
    return params

#%%

import torch.nn.functional as F

class PhysParams(torch.nn.Module):
    def __init__(self, A_init, B_init, Phi_init, Z_init, device=None):
        super().__init__()
        dev = device if device is not None else "cpu"


        self.A_u = torch.nn.Parameter(torch.tensor(float(A_init), device=dev).log())
        self.B_u = torch.nn.Parameter(torch.tensor(float(B_init), device=dev).log())
        self.Z_u = torch.nn.Parameter(torch.tensor(float(Z_init), device=dev).log())


        self.Phi = torch.nn.Parameter(torch.tensor(float(Phi_init), device=dev))

    def forward(self):
        A = F.softplus(self.A_u) + 1e-12
        B = F.softplus(self.B_u) + 1e-12
        Z = F.softplus(self.Z_u) + 1e-12
        Phi = self.Phi
        return A, B, Phi, Z
    

def physics_torch(A, B, Phi, Z, M, F_in):

    term = 1.0 + Z * F_in * M + Z
    base = 1.0 + Z
    return A * (Phi / Z) * (term.pow(1.0 / B) - base.pow(1.0 / B))


def physics_d_torch(A, B, Phi, Z, M, F_in):
    return (A * (Phi * M) / B) * (1.0 + Z + Z * F_in * M).pow(1.0 / B - 1.0)

#%%

class FFNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_layers, activation):
        super().__init__()
        
        layers = []
        

        layers.append(nn.Sequential(nn.Linear(n_input, n_hidden_layers[0]), activation()))

        for i in range(1, len(n_hidden_layers)):
            layers.append(nn.Sequential(nn.Linear(n_hidden_layers[i-1], n_hidden_layers[i]), activation()))
        

        layers.append(nn.Linear(n_hidden_layers[-1], n_output))

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class CNN_Baseline(nn.Module):
    def __init__(self, n_input):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):

        x = x.unsqueeze(1)
        return self.conv(x)


class LSTM_Baseline(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):

        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
def train_baseline_model(model, X_tr, y_tr, X_val, y_val, epochs=1000, lr=1e-4, patience=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_state = None
    counter = 0
    
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = nn.MSELoss()(pred, y_tr)
        loss.backward()
        optimizer.step()
        

        model.eval()
        with torch.no_grad():
            v_loss = nn.MSELoss()(model(X_val), y_val).item()
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        if counter >= patience: break
            
    if best_state: model.load_state_dict(best_state)
    return model

#%%

def fit_affine_ab(p, y):
    ones = torch.ones_like(p)
    Xab = torch.cat([p, ones], dim=1)          
    sol = torch.linalg.lstsq(Xab, y).solution  
    a = sol[0:1]   
    b = sol[1:2]   
    return a.detach(), b.detach()

def pre_shape_to_physics_calibrated(
    model, phys_params,
    X_early, y_early, X_pool,
    fec_col: int, M: float, 
    F_min: float, F_max: float,
    steps: int = 8000, batch_size: int = 256, lr: float = 2e-4,
    w_d1: float = 0.5,  
    w_zero: float = 10.0,  
    device="cuda"
):
    model = model.to(device).train()

    for p in phys_params.parameters():
        p.requires_grad_(False)

    X_early, y_early = X_early.to(device), y_early.to(device)
    X_pool = X_pool.to(device)

    with torch.no_grad():
        F_e = X_early[:, [fec_col]]
        A_t, B_t, Phi_t, Z_t = phys_params()
        M_t = torch.tensor(float(M), device=device)

        p_e = physics_torch(A_t, B_t, Phi_t, Z_t, M_t, F_e)

        a_cal, b_cal = fit_affine_ab(p_e, y_early)
        print(f"Calibration coefficients computed: a = {a_cal.item():.4f}, b = {b_cal.item():.4f}")


    X_zero = X_pool[0:1].clone().detach()
    X_zero[:, fec_col] = 0.0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for it in range(1, steps + 1):
        optimizer.zero_grad()


        idx = torch.randint(0, X_pool.shape[0], (batch_size,), device=device)
        Xb = X_pool[idx].clone()
        Fb = torch.rand(batch_size, 1, device=device) * (F_max - F_min) + F_min
        Xb[:, [fec_col]] = Fb

 
        y_nn = model(Xb)

        p_target = physics_torch(A_t, B_t, Phi_t, Z_t, M_t, Fb).detach()
        y_physics_target = a_cal * p_target + b_cal


        loss_y = (y_nn - y_physics_target).pow(2).mean()
        

        y_zero_pred = model(X_zero)
        loss_z = y_zero_pred.pow(2).mean()

        loss_d1 = torch.tensor(0.0, device=device)
        if w_d1 > 0:
            Xb.requires_grad_(True)
            y_nn_for_grad = model(Xb)

            dy_dX = torch.autograd.grad(y_nn_for_grad.sum(), Xb, create_graph=True)[0]
            dy_dF = dy_dX[:, [fec_col]]
 
            dp_dF = physics_d_torch(A_t, B_t, Phi_t, Z_t, M_t, Fb).detach()
            loss_d1 = (dy_dF - a_cal * dp_dF).pow(2).mean()

 
        
        total_loss = 1.0 * loss_y + w_zero * loss_z + w_d1 * loss_d1
        total_loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print(f"[Pre-shape {it}/{steps}] Total: {total_loss.item():.2e} | Y: {loss_y.item():.2e} | Zero: {loss_z.item():.2e} | D1: {loss_d1.item():.2e}")

    model.eval()
    return model, (a_cal, b_cal)

#%% pre-shaping

def micro_tune_with_early_stopping(
    model, phys_params,
    X_tr, y_tr,
    X_val, y_val,
    fec_col: int, a_cal, b_cal, M,
    max_epochs: int = 2000,
    base_lr: float = 5e-6,
    patience: int = 40,      
    min_delta: float = 1e-8, 
    w_fit: float = 500.0,    
    w_d1: float = 10.0,      
    w_mono: float = 100.0,   
    w_zero: float = 10.0,    
    aug_std: float = 0.01,
    device="cuda"
):

    history = {
        'total': [], 'data': [], 'phys_total': [],
        'l_phys_val': [], 'l_d1': [], 'l_mono': [], 'l_zero': [] 
    }
    
    model = model.to(device)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    M_t = torch.tensor(float(M), device=device)
    a_cal_t = a_cal.to(device)
    b_cal_t = b_cal.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_loss = float('inf')
    best_model_state = None
    stop_counter = 0
    

    F_sampling = torch.linspace(0, 1.2, 512, device=device).view(-1, 1)
    X_sampling = X_tr[0:1].repeat(512, 1)
    X_sampling[:, [fec_col]] = F_sampling

    print(f"🚀 Starting physics-consistent fine-tuning: Max Epochs = {max_epochs}, LR = {base_lr}")

    for ep in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        noise = torch.randn_like(X_tr) * aug_std
        noise[:, fec_col] = 0.0  
        pred_tr = model(X_tr + noise)
        loss_data = (pred_tr - y_tr).pow(2).mean()
        

        Xl = X_sampling.clone().detach().requires_grad_(True)
        yl = model(Xl)
        dy_dX = torch.autograd.grad(yl.sum(), Xl, create_graph=True)[0]
        

        X_zero = X_tr[0:1].clone()
        X_zero[:, fec_col] = 0.0
        loss_zero = model(X_zero).pow(2).mean()
        dy_dF = dy_dX[:, [fec_col]]
        

        loss_mono = torch.relu(-dy_dF).pow(2).mean() 
        

        with torch.no_grad():
            A_t, B_t, Phi_t, Z_t = phys_params()
  
            y_phys_raw = physics_torch(A_t, B_t, Phi_t, Z_t, M_t, Xl[:, [fec_col]])
            y_phys_target = a_cal_t * y_phys_raw + b_cal_t
  
            dp_dF = physics_d_torch(A_t, B_t, Phi_t, Z_t, M_t, Xl[:, [fec_col]])
            target_dF = a_cal_t * dp_dF
        
        loss_phys_val = (yl - y_phys_target).pow(2).mean() 
        loss_d1 = (dy_dF - target_dF).pow(2).mean()       

 
 
        current_phys_loss = w_fit * loss_phys_val + w_d1 * loss_d1 + \
                            w_mono * loss_mono + w_zero * loss_zero

        total_loss = loss_data + current_phys_loss
        
        total_loss.backward()
        optimizer.step()

        history['total'].append(total_loss.item())
        history['data'].append(loss_data.item())
        history['phys_total'].append(current_phys_loss.item())
        history['l_phys_val'].append(loss_phys_val.item())
        history['l_d1'].append(loss_d1.item())
        history['l_mono'].append(loss_mono.item())
        history['l_zero'].append(loss_zero.item())


        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = (val_pred - y_val).pow(2).mean().item()
            
        scheduler.step(val_loss)
        

        if val_loss < (best_loss - min_delta):
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            stop_counter = 0
        else:
            stop_counter += 1
        
        if ep % 200 == 0:
            print(f"Epoch {ep:4d} | Total: {total_loss.item():.2e} | Val Loss: {val_loss:.2e} | d1_Loss: {loss_d1.item():.2e}")

        if stop_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {ep}.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

#%% EOL extrapolation

def get_extrapolated_eol_refined(model, X_sorted, F_sorted, Q_act_sorted, current_M, target_q=0.1, max_cycles=2500):
    model.eval()
    device = next(model.parameters()).device
    from scipy.ndimage import gaussian_filter1d


    Q_act_smooth = gaussian_filter1d(Q_act_sorted, sigma=1.0)
    idx_act = np.where(Q_act_smooth >= target_q)[0]
    actual_eol = F_sorted[idx_act[0]] if len(idx_act) > 0 else F_sorted[-1]

    full_efc_scan = np.linspace(0, max_cycles, 2000)
    

    last_x = X_sorted[[-1]]
    X_scan = last_x.repeat(2000, 1)
    X_scan[:, 0] = torch.tensor(full_efc_scan / current_M).to(device)

    with torch.no_grad():
        Q_pred_raw = model(X_scan).cpu().numpy().flatten()
    
    Q_pred_smooth = gaussian_filter1d(Q_pred_raw, sigma=2.0)

    idx_pre = np.where(Q_pred_smooth >= target_q)[0]
    pred_eol = full_efc_scan[idx_pre[0]] if len(idx_pre) > 0 else max_cycles

    return pred_eol, actual_eol

#%% fast version for time alignment

def micro_tune_phys_adaptive(
    model, phys_params,
    X_tr, y_tr, X_val, y_val,
    fec_col: int, a_cal, b_cal, M,
    max_epochs: int = 2000,
    nn_lr: float = 5e-6,
    phys_lr: float = 5e-7,
    patience: int = 40,
    w_fit: float = 400.0,
    w_d1: float = 15.0,
    w_phys_v: float = 10.0,
    w_mono: float = 500.0,
    w_zero: float = 20.0,
    aug_std: float = 0.02,
    device="cuda"
):

    model = model.to(device)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    M_t = torch.tensor(float(M), device=device)
    a_cal_t = a_cal.to(device)
    b_cal_t = b_cal.to(device)


    with torch.no_grad():
        A0, B0, Phi0, Z0 = phys_params()

    param_history = {'A': [], 'B': [], 'Phi': [], 'Z': []}
    for p in phys_params.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': nn_lr},
        {'params': phys_params.parameters(), 'lr': phys_lr} 
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    best_loss, best_model_state, best_phys_state = float('inf'), None, None
    stop_counter = 0


    pure_duration = 0
   

    for ep in range(1, max_epochs + 1):
        
        epoch_start = time.time()
        
        model.train()
        optimizer.zero_grad()
        

        F_sampling = torch.linspace(0, 1.2, 512, device=device).view(-1, 1).requires_grad_(True)
        X_sampling = X_tr[0:1].repeat(512, 1).detach()
        X_sampling[:, [fec_col]] = F_sampling 
        

        A_curr, B_curr, Phi_curr, Z_curr = phys_params()
        

        noise = torch.randn_like(X_tr) * aug_std
        noise[:, fec_col] = 0.0
        loss_fit = (model(X_tr + noise) - y_tr).pow(2).mean()
        

        X_zero = X_tr[0:1].clone()
        X_zero[:, fec_col] = 0.0
        loss_zero = model(X_zero).pow(2).mean()
        

        yl = model(X_sampling)
        dy_dF = torch.autograd.grad(yl.sum(), F_sampling, create_graph=True)[0]
        
        y_phys_target = a_cal_t * physics_torch(A_curr, B_curr, Phi_curr, Z_curr, M_t, F_sampling) + b_cal_t
        target_dF = a_cal_t * physics_d_torch(A_curr, B_curr, Phi_curr, Z_curr, M_t, F_sampling)
        
        loss_d1 = (dy_dF - target_dF).pow(2).mean()
        loss_phys_v = (yl - y_phys_target).pow(2).mean()
        loss_mono = torch.relu(-dy_dF).pow(2).mean()

        total_loss = (w_fit * loss_fit + w_d1 * loss_d1 + 
                      w_phys_v * loss_phys_v + w_mono * loss_mono + w_zero * loss_zero)
        
        total_loss.backward()
        optimizer.step()
        

        pure_duration += (time.time() - epoch_start)
        

        if ep % 10 == 0:
            with torch.no_grad():
                At, Bt, Phit, Zt = phys_params()

                param_history['A'].append(((At - A0) / A0 * 100).item())
                param_history['B'].append(((Bt - B0) / B0 * 100).item())
                param_history['Phi'].append(((Phit - Phi0) / Phi0 * 100).item())
                param_history['Z'].append(((Zt - Z0) / Z0 * 100).item())


        model.eval()
        with torch.no_grad():
            val_loss = (model(X_val) - y_val).pow(2).mean().item()
        
        scheduler.step(val_loss)
        if val_loss < (best_loss - 1e-8):
            best_loss = val_loss
      
            best_model_state = copy.deepcopy(model.state_dict())
            best_phys_state = copy.deepcopy(phys_params.state_dict())
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter >= patience:
            break


    if best_model_state:
        model.load_state_dict(best_model_state)
        phys_params.load_state_dict(best_phys_state)
    

    return model, phys_params, param_history, pure_duration

#%% NN vs. physics

def validate_pinn_physics_flat(
    model, phys_params, X_src_all, M, a_cal, b_cal,
    device="cuda", fec_col=0
):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import matplotlib as mpl

    model.eval()

    X_real = X_src_all.clone().detach().to(device)
    X_real.requires_grad_(True)

    F_norm = X_real[:, [fec_col]]
    y_nn = model(X_real)

    dy_dX = torch.autograd.grad(
        y_nn.sum(),
        X_real,
        create_graph=False,
        retain_graph=False
    )[0]

    dy_dF_nn = dy_dX[:, [fec_col]]

    with torch.no_grad():
        A, B, Phi, Z = phys_params()
        M_t = torch.tensor(float(M), device=device)

        y_phys_raw = physics_torch(A, B, Phi, Z, M_t, F_norm)
        dy_phys_raw = physics_d_torch(A, B, Phi, Z, M_t, F_norm)

        y_phys_target = a_cal * y_phys_raw + b_cal
        dy_phys_target = a_cal * dy_phys_raw

    F = F_norm.detach().cpu().numpy().flatten()
    y_nn_np = y_nn.detach().cpu().numpy().flatten()
    dy_nn_np = dy_dF_nn.detach().cpu().numpy().flatten()
    y_phys_np = y_phys_target.detach().cpu().numpy().flatten()
    dy_phys_np = dy_phys_target.detach().cpu().numpy().flatten()

    idx = np.argsort(F)

  
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 2.8),   
        dpi=300
    )

    C_NN = "#4A90E2"
    C_PHYS = "#D62728"
    C_SLOPE_NN = "#F5A623"
    C_SLOPE_PHYS = "#2CA02C"


    ax = axes[0]
    ax.scatter(
        F[idx],
        y_nn_np[idx],
        s=10,
        alpha=0.50,
        color=C_NN,
        label="NN"
    )
    ax.plot(
        F[idx],
        y_phys_np[idx],
        lw=2.2,
        linestyle="--",
        color=C_PHYS,
        label="Physics"
    )
    ax.set_ylabel(r"$Q_{loss}$", labelpad=6)

   
    ax = axes[1]
    ax.scatter(
        F[idx],
        dy_nn_np[idx],
        s=10,
        alpha=0.50,
        color=C_SLOPE_NN,
        label="NN"
    )
    ax.plot(
        F[idx],
        dy_phys_np[idx],
        lw=2.2,
        linestyle="--",
        color=C_SLOPE_PHYS,
        label="Physics"
    )
    ax.set_ylabel(r"$\frac{dy}{dEFC_{norm}}$", labelpad=6)

    for ax in axes:
        ax.set_xlabel(r"$Normalized\ EFC\ (EFC/M_{phys})$", labelpad=6)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.tick_params(
            direction="out",
            length=4,
            width=1.0,
            labelsize=13
        )

        ax.grid(False)

        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.75,
            edgecolor="none"
        )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.25,
        top=0.92,
        wspace=0.30
    )

    plt.show()

    mpl.rcParams.update(mpl.rcParamsDefault)

#%%

def plot_Q_ALT_transfer(
    F_tr, Q_tr,    
    F_te, Q_te,      
    F_hat, Q_hat,    
    condition=None,  
    file_suffix=None, 
    ratio_early=None
):
    plt.figure(figsize=(12/2.52, 7.5/2.52))
    plt.rcParams.update({'font.size': 12})


    plt.scatter(F_tr, Q_tr*100, s=12, alpha=0.8,
                color='tab:orange', label='Visible Data (Known)', zorder=3)
    plt.scatter(F_te, Q_te*100, s=10, alpha=0.4,
                color='grey', label='Future Truth (Hidden)', zorder=1)


    order = np.argsort(F_hat)
    plt.scatter(np.array(F_hat)[order], np.array(Q_hat)[order]*100, s=6, color='tab:blue')


    plt.xlabel('EFC')
    plt.ylabel(r'$Q_{loss}[\%]$')


    if ratio_early is not None:
        title_str = f"Target transfer: Ratio={ratio_early:.2f}"
    else:
        title_str = "Target transfer"

    plt.title(title_str, fontsize=13, pad=10)
    
    plt.legend(fontsize=10, frameon=True, framealpha=0.85, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

#%% load data file

def load_cells_to_dict(folder, cell_list):
    all_data = {}

    unique_cells = set(cell_list)
    print(f"📦 Total cells requested: {len(cell_list)} (Unique: {len(unique_cells)})")
    
    for cell_id in unique_cells:
        file_path = os.path.join(folder, f"{cell_id}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                all_data[cell_id] = pickle.load(f)
            print(f"  ✅ Loaded [{cell_id}] successfully.")
        else:
            print(f"  ❌ Error: {file_path} not found.")
            
    return all_data

data_folder = 'sourcedata' 


cell_ids = [
    'cell01', 'cell02', 'cell01', 'cell03', 
    'cell04', 'cell05', 'cell01', 'cell08', 
    'cell06', 'cell07', 'cell09', 'cell10', 
    'cell11', 'cell12', 'cell13', 'cell14', 
    'cell15', 'cell16', 'cell17', 'cell18'
]


data = load_cells_to_dict(data_folder, cell_ids)

#%%  Physical Loss, Adaptive Aging Parameter Updates, and Comparison Between NN Predictions and Physical Values (Figures 2a, 2b, and 2d)

config_file_path = os.path.join("sourcedata", "pipeline_config.json")

with open(config_file_path, "r", encoding="utf-8") as f:
    loaded_config = json.load(f)


cell_pairs = [tuple(pair) for pair in loaded_config["cell_pairs"]]
param_grid = {k: tuple(v) for k, v in loaded_config["param_grid"].items()} 
PARAM_MAP  = loaded_config["PARAM_MAP"]

print(f"🎉 Experiment configuration loaded successfully! Successfully initialized {len(cell_pairs)} battery pairs and the adaptive physical hyperparameter matrix.。")

def generate_reproducible_seeds(master_seed=2026, num_seeds=2):

    random.seed(master_seed)
    seeds = random.sample(range(10000, 99999), num_seeds)
    print(f"✅ Ten experimental seeds generated from master seed {master_seed}: {seeds}")
    return seeds


def check_physical_integrity(model,X_check, Q_check, M, device): 
    """
    Assessing the model's physical consistency through high-density sampling and gradient analysis.
    """
    model.eval()
    with torch.no_grad():

        Y_te = model(X_check)

        errors = torch.abs(Y_te - Q_check)

        mean_err = errors.mean().item()
    
        std_err = errors.std().item()
    
    with torch.enable_grad():
     
        f_v = torch.linspace(0, 1.25, 300, device=device).view(-1, 1)
        last_feat = X_check[[-1], 1:].detach()
        X_v = torch.cat([f_v, last_feat.expand(300, -1)], dim=1).requires_grad_(True)
        

        Y_v = model(X_v)
        grad_v = torch.autograd.grad(
            outputs=Y_v, inputs=X_v,
            grad_outputs=torch.ones_like(Y_v),
            create_graph=False
        )[0]
        

        slopes_v = grad_v[:, 0] / M

        abs_min_v = slopes_v.min().item()  
        avg_slope_v = slopes_v.mean().item() 
        std_slope_v = slopes_v.std().item()
        cv_slope_v = std_slope_v / (abs(avg_slope_v) + 1e-9)
        min_y_v = Y_v.min().item()
   
        X_te_grad = X_check.clone().detach().requires_grad_(True)
        Y_te = model(X_te_grad)
        grad_te = torch.autograd.grad(Y_te, X_te_grad, torch.ones_like(Y_te))[0]
        slopes_te = grad_te[:, 0] / M
        
        abs_min_te = slopes_te.min().item()  
        min_y_te = Y_te.min().item()         
        
  
    
        no_negative = (min_y_v > 1e-5) and (min_y_te > 1e-5)
        

        is_monotonic = (abs_min_v > -1e-4) and (abs_min_te > -1e-4)
        

        is_smooth = (cv_slope_v < 0.8) and (std_slope_v < 0.05)
        

        is_reasonable = (0 < avg_slope_v < 0.5)
        
        residual_ok = (mean_err < 0.05) and (std_err < 0.03)
        

        is_safe = no_negative and is_monotonic and is_smooth and is_reasonable and residual_ok
        
        diag_info = (f"MinY_Te: {min_y_te:.3f}, MinSlope_Te: {abs_min_te:.5f}, "
                     f"CV_V: {cv_slope_v:.3f}, Std_V: {std_slope_v:.4f}")
        
        return is_safe, diag_info
        
    

random_seeds = generate_reproducible_seeds(2026, 1)

model_names_extended = [
    'PINN_Base', 'PINN_Phys', 
    'FNN_Base', 'FNN_Phys', 
    'CNN_Base', 'CNN_Phys', 
    'LSTM_Base', 'LSTM_Phys'
]

RATIO_EARLY = 0.02  

parity_data = {
    'Source Train': {'actual': [], 'pred': []},
    'Source Val':   {'actual': [], 'pred': []},
    'Target Train': {'actual': [], 'pred': []},
    'Target Test':  {'actual': [], 'pred': []}
}

seed_results = {name: {i: {'actual': [], 'pred': []} for i in range(len(random_seeds))} 
                for name in model_names_extended}



seed_eol_absolute = {name: {i: {'pred_eol': [], 'actual_eol': []} for i in range(len(random_seeds))} 
                    for name in model_names_extended}


training_times = {name: [] for name in model_names_extended}


landscape_snapshots = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transfer_results = {}

parity_results = {name: {'actual': [], 'pred': []} for name in model_names_extended}
random_seeds = generate_reproducible_seeds(2026, 10)

model_names_extended = [
    'PINN_Base', 'PINN_Phys', 
    'FNN_Base', 'FNN_Phys', 
    'CNN_Base', 'CNN_Phys', 
    'LSTM_Base', 'LSTM_Phys'
]


seed_results = {name: {i: {'actual': [], 'pred': [], 'type': []} 
                for i in range(len(random_seeds))} for name in model_names_extended}



seed_eol_absolute = {name: {i: {'pred_eol': [], 'actual_eol': [], 'cell_id': [], 'type': []} 
                    for i in range(len(random_seeds))} for name in model_names_extended}


training_times = {name: [] for name in model_names_extended}


landscape_snapshots = {i: {} for i in range(len(random_seeds))}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transfer_results = {}

parity_results = {name: {'actual': [], 'pred': []} for name in model_names_extended}

for s_idx, seed_val in enumerate(random_seeds):
    

    if s_idx > 0:
        print("\n🛑 [Pipeline Interceptor] First random seed completed. Remaining seed iterations are adaptively skipped to reduce computational cost.")
        break
    
    print(f"\n" + "🌟" * 20)
    print(f"🚀 Running experiment {s_idx+1}/10 (Seed: {seed_val})")
    print("🌟" * 20)
    

    target_last_pair = cell_pairs[:1]
    
    for src_id, tgt_id in target_last_pair:

        cfg = PARAM_MAP.get(tgt_id, PARAM_MAP['default'])
        current_epochs, current_lr = param_grid.get(tgt_id, (500, 5e-6))

        experiment_tracks = [
            {'name': 'Base', 'data_source': data},
        ]
        
        for track in experiment_tracks:
            mode = track['name']
            d_src = track['data_source']
            pinn_id = f"PINN_{mode}"
            
  
            success_phys = False
            max_retries = 3 
            retry_count = 0
       
            current_loop_seed = seed_val
            
            while retry_count < max_retries and not success_phys:               
            
       
                set_seed(seed_val)
                

                source_prepared, shared_features = prepare_cell_dataframe_transfer(
                    d_src[src_id], N_total=16, use_phys=False
                )
                target_prepared, _ = prepare_cell_dataframe_transfer(
                    d_src[tgt_id], fixed_features=shared_features
                )
                
                tgt_tr_df, tgt_val_df, tgt_te_df = split_data_for_transfer(
                    target_prepared, is_source=False, ratio_early=RATIO_EARLY, ratio_val=0.2
                )
                src_train_df, src_val_df, _ = split_data_for_transfer(source_prepared, is_source=True, ratio_val=0.2)
                

                src_pack, tgt_pack, M, scaler = scale_transfer_flow(src_train_df, src_val_df, tgt_tr_df, tgt_val_df, tgt_te_df, M_fixed=1200)
                X_src_tr, Q_src_tr, X_src_val, Q_src_val = [t.to(device) for t in src_pack]
                X_tgt_tr, Q_tgt_tr, X_tgt_val, Q_tgt_val, X_tgt_te, Q_tgt_te = [t.to(device) for t in tgt_pack]
        
  
                pinn_id = f"PINN_{mode}"
                set_seed(seed_val)
                

                F_src_raw = (torch.cat([X_src_tr[:, 0], X_src_val[:, 0]]) * M).cpu().numpy()
                Q_src_raw = torch.cat([Q_src_tr, Q_src_val]).cpu().numpy().ravel()
                
                X_src_tr, Q_src_tr = X_src_tr.to(device), Q_src_tr.to(device)
                X_src_val, Q_src_val = X_src_val.to(device), Q_src_val.to(device)
                X_tgt_tr, Q_tgt_tr = X_tgt_tr.to(device), Q_tgt_tr.to(device)
                X_tgt_val, Q_tgt_val = X_tgt_val.to(device), Q_tgt_val.to(device)
                X_tgt_te, Q_tgt_te = X_tgt_te.to(device), Q_tgt_te.to(device)

                A_s, B_s, Phi_s, Z_s = fit_physics_params(F_src_raw, Q_src_raw, M)
                phys_params = PhysParams(A_s, B_s, Phi_s, Z_s, device=device)
                

                F_max_dynamic = float(X_src_val[:, 0].max().item()) * 1.1 

                X_pool_transfer = torch.cat([X_src_tr, X_src_val], dim=0)

                set_seed(seed_val)
                
                model_src = FFNN(n_input=X_src_tr.shape[1], n_output=1, n_hidden_layers=[16, 16], activation=nn.SiLU).to(device)

                model_src, (a_s, b_s) = pre_shape_to_physics_calibrated(
                    model=model_src, phys_params=phys_params,
                    X_early=X_src_tr, y_early=Q_src_tr, X_pool=X_pool_transfer,
                    fec_col=0, M=M, F_min=0.0, F_max=F_max_dynamic,
                    steps=10000, batch_size=256, lr=cfg['pre_lr'], 
                    w_d1=cfg['pre_wd1'], w_zero=cfg['pre_wzero'], 
                    device=device
                )
        
                
                print(f"--- Fine-tuning source battery {src_id} (adaptive mode) ---")
                model_src, history_src = micro_tune_with_early_stopping(
                    model=model_src, 
                    phys_params=phys_params,
                    X_tr=X_src_tr, y_tr=Q_src_tr,
                    X_val=X_src_val, y_val=Q_src_val,
                    fec_col=0, a_cal=a_s, b_cal=b_s, M=M,
                    max_epochs=5000,  
                    base_lr=1e-5,      
                    patience=80,       
                    w_fit=500.0, 
                    w_d1=cfg['src_wd1'], 
                    w_mono=cfg['src_wmono'], 
                    aug_std=cfg['src_aug'], 
                    device=device
                )

                plot_physics_loss_1x3_compact(history_src)

                model_src.eval()
                with torch.no_grad():
              
                    X_src_full = torch.cat([X_src_tr, X_src_val], dim=0)
                    Q_src_full_act = torch.cat([Q_src_tr, Q_src_val]).cpu().numpy().flatten()
                    F_src_full_efc = X_src_full[:, 0].cpu().numpy() * M
                    
                    s_sort = np.argsort(F_src_full_efc)
                    
                    pair_key = f"Pair_{src_id}_to_{tgt_id}"
                    
                    transfer_results[f"{pair_key}_Src"] = {
                        'efc': F_src_full_efc[s_sort],
                        'actual': Q_src_full_act[s_sort],
                        'predicted': model_src(X_src_full[s_sort]).cpu().numpy().flatten(),
                        'type': 'Source'
                    }
                    

                print(f"\n🧪 Diagnosing the physics learning performance of source battery {src_id}...")
                
     
                validate_pinn_physics_flat(
                    model=model_src, 
                    phys_params=phys_params, 
                    X_src_all=X_src_full, 
                    M=M, 
                    a_cal=a_s,          
                    b_cal=b_s,           
                    device=device,
                    fec_col=0
                )
                
                

                F_src_tr_raw = (X_src_tr[:, 0] * M).cpu().numpy()
                Q_src_tr_raw = Q_src_tr.cpu().numpy().flatten()
        
                F_src_val_raw = (X_src_val[:, 0] * M).cpu().numpy()
                Q_src_val_raw = Q_src_val.cpu().numpy().flatten()
        
                

                with torch.no_grad():
                    parity_data['Source Train']['actual'].extend(Q_src_tr.cpu().numpy().flatten())
                    parity_data['Source Train']['pred'].extend(model_src(X_src_tr).cpu().numpy().flatten())
                    
                    parity_data['Source Val']['actual'].extend(Q_src_val.cpu().numpy().flatten())
                    parity_data['Source Val']['pred'].extend(model_src(X_src_val).cpu().numpy().flatten())
                    

                    y_p_src = model_src(X_src_full).cpu().numpy().flatten()
                    seed_results[pinn_id][s_idx]['actual'].extend(Q_src_full_act)
                    seed_results[pinn_id][s_idx]['pred'].extend(y_p_src)
                    seed_results[pinn_id][s_idx]['type'].extend(['Source'] * len(Q_src_full_act))
                

                    F_src_efc = X_src_full[:, 0].cpu().numpy() * M
                    s_idx_src = np.argsort(F_src_efc)
                    p_eol_src, a_eol_src = get_extrapolated_eol_refined(
                        model=model_src, X_sorted=X_src_full[s_idx_src], 
                        F_sorted=F_src_efc[s_idx_src], Q_act_sorted=Q_src_full_act[s_idx_src],
                        current_M=M, target_q=0.1
                    )
                    seed_eol_absolute[pinn_id][s_idx]['pred_eol'].append(p_eol_src)
                    seed_eol_absolute[pinn_id][s_idx]['actual_eol'].append(a_eol_src)
                    seed_eol_absolute[pinn_id][s_idx]['cell_id'].append(src_id)
                    seed_eol_absolute[pinn_id][s_idx]['type'].append('Source')
                    
        

                model_tgt = copy.deepcopy(model_src).to(device)
                phys_params_tgt = copy.deepcopy(phys_params).to(device)
                             

                with torch.no_grad():
                    F_tgt_e = X_tgt_tr[:, [0]].to(device)
                    A_t, B_t, Phi_t, Z_t = phys_params()
                    p_tgt_e = physics_torch(A_t, B_t, Phi_t, Z_t, torch.tensor(M).to(device), F_tgt_e * M)
                    a_tgt, b_tgt = fit_affine_ab(p_tgt_e, Q_tgt_tr.to(device))
        

                print(f"🚀 Starting physics-adaptive fine-tuning for {tgt_id}...")

                # ==========================================
                model_tgt, phys_params_tgt, param_hist, tune_duration = micro_tune_phys_adaptive(
                    model=model_tgt, 
                    phys_params=phys_params_tgt,
                    X_tr=X_tgt_tr, 
                    y_tr=Q_tgt_tr,
                    X_val=X_tgt_val, 
                    y_val=Q_tgt_val,
                    fec_col=0, 
                    a_cal=a_tgt, 
                    b_cal=b_tgt, 
                    M=M,
                    max_epochs=current_epochs,
                    nn_lr=current_lr, 
                    phys_lr=1e-5,    #
                    patience=cfg.get('tgt_patience', 40),
                    w_fit=cfg['tgt_wfit'],    
                    w_d1=cfg['tgt_wd1'],      
                    w_phys_v=cfg['tgt_wphys_v'],  
                    w_zero=cfg['tgt_wzero'],     
                    device=device
                )
                
 
                training_times[pinn_id].append(tune_duration)  
                
                is_safe, diag = check_physical_integrity(
                    model=model_tgt, 
                    X_check=X_tgt_val,  
                    Q_check=Q_tgt_val,  
                    M=M, 
                    device=device
                )
                
                
                if mode == 'Base':
                    success_phys = True
                    if not is_safe:
                        print(f"      ⚠️  Base model violates physical constraints ({diag}); retained as a negative example.")

                
            # ==========================================
            # Physical Parameter Evolution Diagnostics for the Target Battery
            # ==========================================
            print(f"\n🧪 Generating a comprehensive parameter evolution diagnostic plot for target battery {tgt_id}...")
            plot_physics_parameter_evolution_1x4_final(param_hist)
            # --------------------------------------------------------
            model_tgt.eval()
            with torch.no_grad():
                              
                X_tgt_full = torch.cat([X_tgt_tr, X_tgt_val, X_tgt_te], dim=0)
  
                Q_tgt_full_act = torch.cat([Q_tgt_tr, Q_tgt_val, Q_tgt_te]).cpu().numpy().flatten()
                F_tgt_full_efc = X_tgt_full[:, 0].cpu().numpy() * M         
   
                sort_idx = np.argsort(F_tgt_full_efc)
                F_sorted = F_tgt_full_efc[sort_idx]
                Q_act_sorted = Q_tgt_full_act[sort_idx]
                X_sorted = X_tgt_full[sort_idx]
                
                Q_raw = model_tgt(X_sorted).cpu().numpy().flatten()
                Q_pred_smooth = gaussian_filter1d(Q_raw, sigma=3.0)
                
                parity_results[pinn_id]['actual'].extend(Q_act_sorted)
                parity_results[pinn_id]['pred'].extend(Q_raw)            

                F_known = (torch.cat([X_tgt_tr[:, 0], X_tgt_val[:, 0]]) * M).cpu().numpy()
                Q_known = torch.cat([Q_tgt_tr, Q_tgt_val]).cpu().numpy().flatten()
                
                F_unknown = (X_tgt_te[:, 0] * M).cpu().numpy()
                Q_unknown = Q_tgt_te.cpu().numpy().flatten()
                
                plot_Q_ALT_transfer(
                    F_tr=F_known, 
                    Q_tr=Q_known,
                    F_te=F_unknown, 
                    Q_te=Q_unknown,
                    F_hat=F_sorted,  
                    Q_hat=Q_pred_smooth,
                    ratio_early=RATIO_EARLY  
                )

                pred_eol_val, act_eol_val = get_extrapolated_eol_refined(
                    model=model_tgt, 
                    X_sorted=X_sorted,      
                    F_sorted=F_sorted,     
                    Q_act_sorted=Q_act_sorted,
                    current_M=M, 
                    target_q=0.1
                )

                seed_eol_absolute[pinn_id][s_idx]['pred_eol'].append(pred_eol_val)
                seed_eol_absolute[pinn_id][s_idx]['actual_eol'].append(act_eol_val)
                seed_eol_absolute[pinn_id][s_idx]['cell_id'].append(tgt_id)
                seed_eol_absolute[pinn_id][s_idx]['type'].append('Target') 

            m_display_name = "PINN_Base" if mode == 'Base' else "PINN_Phys"
            snapshot_key = f"{tgt_id}_{m_display_name}"

            landscape_snapshots[s_idx][snapshot_key] = {
                'model_state': copy.deepcopy(model_tgt.state_dict()),
                'phys_params': copy.deepcopy(phys_params_tgt.state_dict()) if mode == 'Phys' else None,
                'X_val': X_tgt_val.clone(),
                'y_val': Q_tgt_val.clone(),
                'M': M,
                'X_sorted_full': X_sorted.clone(),
                'feature_names': target_prepared.columns.tolist()
            }               
            
            
            pinn_id = f"PINN_{mode}"

            # --- [PINN results] ---
            if success_phys:
                model_tgt.eval()
                with torch.no_grad():
                    y_p = model_tgt(X_tgt_te).cpu().numpy().flatten()
                    y_t = Q_tgt_te.cpu().numpy().flatten()
                    seed_results[pinn_id][s_idx]['actual'].extend(y_t)
                    seed_results[pinn_id][s_idx]['pred'].extend(y_p)
            else:
                print(f"      ❌ {tgt_id} ({mode}) still failed after retries. Skipping this seed.")
            

            baselines_map = {
                f'FNN_{mode}': FFNN(X_src_tr.shape[1], 1, [16, 16], activation=nn.SiLU).to(device),
                f'CNN_{mode}': CNN_Baseline(X_src_tr.shape[1]).to(device),
                f'LSTM_{mode}': LSTM_Baseline(X_src_tr.shape[1]).to(device)
            }

            for b_name, b_model in baselines_map.items():
                set_seed(seed_val)
                b_start_t = time.time()
                
                b_model = train_baseline_model(b_model, X_src_tr, Q_src_tr, X_src_val, Q_src_val, epochs=1000)
                b_model = train_baseline_model(b_model, X_tgt_tr, Q_tgt_tr, X_tgt_val, Q_tgt_val, epochs=current_epochs)
                

                b_duration = time.time() - b_start_t
                training_times[b_name].append(b_duration)
                
                b_model.eval()
                with torch.no_grad():
                    y_p = b_model(X_tgt_te).cpu().numpy().flatten()
                    seed_results[b_name][s_idx]['actual'].extend(y_t)
                    seed_results[b_name][s_idx]['pred'].extend(y_p)                           
                    b_pred_eol, b_act_eol = get_extrapolated_eol_refined(
                        model=b_model, 
                        X_sorted=X_sorted,     
                        F_sorted=F_sorted,     
                        Q_act_sorted=Q_act_sorted,
                        current_M=M, 
                        target_q=0.1
                    )
                    
                    seed_eol_absolute[b_name][s_idx]['pred_eol'].append(b_pred_eol)
                    seed_eol_absolute[b_name][s_idx]['actual_eol'].append(b_act_eol)

                snapshot_key = f"{tgt_id}_{b_name}"
                
                landscape_snapshots[s_idx][snapshot_key] = {
                    'model_state': copy.deepcopy(b_model.state_dict()),
                    'phys_params': None,
                    'X_val': X_tgt_val.clone(),
                    'y_val': Q_tgt_val.clone(),
                    'M': M,
                    'X_sorted_full': X_sorted.clone(),
                    'feature_names': target_prepared.columns.tolist()
                }
                
#%% load data results

import torch
import os

save_dir = "Saved_Results_2026"
pt_path = os.path.join(save_dir, "landscape_snapshots.pt")

print("=== 🔍 Recovering core Loss Landscape assets... ===")
if os.path.exists(pt_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    landscape_snapshots = torch.load(pt_path, map_location=device)
    print(f"✅ Snapshot loaded successfully!")
    

    first_seed = list(landscape_snapshots.keys())[0]
    available_models = list(landscape_snapshots[first_seed].keys())
    print(f"📌 Model snapshot fingerprints available for seed [{first_seed}]:")
    for k in available_models:
        print(f"  - {k}")
else:
    raise FileNotFoundError(f"❌ Snapshot file not found: {pt_path}")

#%% Loss Landscape Comparison of Four ML Models for Extrapolation Performance Evaluation (Figure 2d)

# function need
def create_model_by_name(m_name, input_dim):
    if 'CNN' in m_name:
        return CNN_Baseline(input_dim)
    elif 'LSTM' in m_name:
        return LSTM_Baseline(input_dim)
    else:
        return FFNN(n_input=input_dim, n_output=1, n_hidden_layers=[16, 16], activation=nn.SiLU)
    

class FFNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_layers, activation):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Sequential(nn.Linear(n_input, n_hidden_layers[0]), activation()))
        
        for i in range(1, len(n_hidden_layers)):
            layers.append(nn.Sequential(nn.Linear(n_hidden_layers[i-1], n_hidden_layers[i]), activation()))

        layers.append(nn.Linear(n_hidden_layers[-1], n_output))

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    

class CNN_Baseline(nn.Module):
    def __init__(self, n_input):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):

        x = x.unsqueeze(1)
        return self.conv(x)


class LSTM_Baseline(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):

        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])    


def get_landscape_data_extrapolation(model, snap, X_test, y_test, range_val=0.4, steps=30, fixed_dirs=None):

    model.load_state_dict(snap['model_state'])
    model.eval()
    weights = [p.data.clone() for p in model.parameters()]
    

    def generate_random_dir(w_list):
        dirs = [torch.randn_like(p) for p in w_list]
        norm = torch.sqrt(sum([d.pow(2).sum() for d in dirs]))
        return [d / norm for d in dirs]

    dir_x, dir_y = [], []

    if fixed_dirs is None:
        dir_x = generate_random_dir(weights)
        dir_y = generate_random_dir(weights)
    else:
        old_dx_list, old_dy_list = fixed_dirs

        if len(old_dx_list) != len(weights) or old_dx_list[0].shape != weights[0].shape:

            dir_x = generate_random_dir(weights)
            dir_y = generate_random_dir(weights)
        else:

            dir_x, dir_y = old_dx_list, old_dy_list

    coords = np.linspace(-range_val, range_val, steps)
    X, Y = np.meshgrid(coords, coords)
    Z = np.zeros((steps, steps))

    X_test_dev = X_test.to(next(model.parameters()).device)
    y_test_dev = y_test.to(next(model.parameters()).device)

    with torch.no_grad():
        for i in range(steps):
            for j in range(steps):

                for p, orig, dx, dy in zip(model.parameters(), weights, dir_x, dir_y):
                    p.data = orig + X[i, j] * dx + Y[i, j] * dy
                
                pred = model(X_test_dev)
                loss = torch.mean((pred - y_test_dev)**2).item()

                Z[i, j] = np.log10(loss + 1e-9)


    model.load_state_dict(snap['model_state'])
    return X, Y, Z, (dir_x, dir_y)

def plot_four_landscape_extrapolation_final(data_list):


    plt.rcParams.update(plt.rcParamsDefault)

    TITLE_SIZE = 32
    LABEL_SIZE = 26
    TICK_SIZE = 24
    CBAR_SIZE = 24

    fig = plt.figure(figsize=(28, 8.5), dpi=150)

    all_z = [d["Z"] for d in data_list]
    global_z_min = min(z.min() for z in all_z)
    global_z_max = max(z.max() for z in all_z)

    title_map = {
        "PINN_Base": "PINN",
        "FNN_Base": "FNN",
        "CNN_Base": "CNN",
        "LSTM_Base": "LSTM"
    }

    plot_width = 0.18
    ax_positions = [
        [0.07, 0.14, plot_width, 0.74],
        [0.29, 0.14, plot_width, 0.74],
        [0.51, 0.14, plot_width, 0.74],
        [0.73, 0.14, plot_width, 0.74],
    ]

    main_cmap = "plasma"
    last_surf = None

    for i, (data, pos) in enumerate(zip(data_list, ax_positions)):
        ax = fig.add_axes(pos, projection="3d")

        surf = ax.plot_surface(
            data["X"],
            data["Y"],
            data["Z"],
            cmap=main_cmap,
            vmin=global_z_min,
            vmax=global_z_max,
            edgecolor="none",
            alpha=0.90,
            antialiased=True
        )

        last_surf = surf

        ax.view_init(elev=35, azim=-120)
        ax.set_zlim(global_z_min, global_z_max)

        ax.set_xlabel(r"Dir. $\delta$", fontsize=LABEL_SIZE, labelpad=24)
        ax.set_ylabel(r"Dir. $\eta$", fontsize=LABEL_SIZE, labelpad=24)

        title = title_map.get(data["name"], data["name"])
        ax.set_title(title, fontsize=TITLE_SIZE, pad=4)

        ax.xaxis.set_major_locator(MultipleLocator(0.4))
        ax.yaxis.set_major_locator(MultipleLocator(0.4))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))

        ax.tick_params(axis="x", labelsize=TICK_SIZE, pad=10)
        ax.tick_params(axis="y", labelsize=TICK_SIZE, pad=10)

        ax.tick_params(axis="z", labelsize=TICK_SIZE, pad=18)

        try:
            ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0
            ax.zaxis._axinfo["tick"]["outward_factor"] = 0.35
        except Exception:
            pass

    cax = fig.add_axes([0.94, 0.30, 0.010, 0.42])
    cbar = fig.colorbar(last_surf, cax=cax)

    cb_ticks = np.linspace(global_z_min, global_z_max, 4)
    cbar.set_ticks(cb_ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in cb_ticks])
    cbar.ax.tick_params(labelsize=CBAR_SIZE, width=1.2, length=5)

    fig.text(
        0.035,
        0.52,
        r"$\log_{10}(MSE)$",
        rotation=90,
        fontsize=28,
        va="center",
        ha="center"
    )

    plt.show()

def seed_everything_for_landscape(seed=2026):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

last_seed_idx = len(random_seeds) - 1  
print(f"📡 Extracting data from the final experiment (Seed Index: {last_seed_idx}) for loss landscape analysis...")


current_seed_snapshots = landscape_snapshots[last_seed_idx]


target_cell = 'cell18' 
models_to_draw = ['PINN_Base', 'FNN_Base', 'CNN_Base', 'LSTM_Base']
landscape_extrap_list = []


seed_everything_for_landscape(1996)

X_extrap = X_tgt_te 
y_extrap = Q_tgt_te

base_dirs = None

for m_name in models_to_draw:
    key = f"{target_cell}_{m_name}"
    

    if key in current_seed_snapshots:
        print(f"🚀 Mapping the loss landscape of {m_name} under extrapolation conditions...")
        snap = current_seed_snapshots[key] 
        
        input_dim = snap['X_val'].shape[1]
        
        temp_model = create_model_by_name(m_name, input_dim).to(device)
        

        X, Y, Z, dirs = get_landscape_data_extrapolation(
            temp_model, snap, X_extrap, y_extrap, 
            range_val=0.4, steps=35, fixed_dirs=base_dirs 
        )
        

        if base_dirs is None: base_dirs = dirs
        
        landscape_extrap_list.append({'X': X, 'Y': Y, 'Z': Z, 'name': m_name})
    else:
        print(f"❌ Warning: Snapshot {key} not found in Seed {last_seed_idx}.")

if len(landscape_extrap_list) > 0:
    plot_four_landscape_extrapolation_final(landscape_extrap_list)
else:
    print("❌ Error: No valid loss landscape data available for plotting.")
    
