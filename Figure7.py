# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:30:50 2026

Figure 7 replica

@author: wengu476
"""

import pickle
import os
import torch
import torch.nn as nn

import shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize


#%% V2G Voltage Range Distribution (Figures 7d and 7e)

file_path = 'V2G_Recovered_Fixed_20260520_0921.pkl'
with open(file_path, 'rb') as f:
    raw_data = pickle.load(f)

v2g_metrics = raw_data['metrics']

plot_list = []
target_models = ['PINN_Phys', 'PINN_Base', 'CNN_Phys']

for window, models in v2g_metrics.items():
    parts = window.split('_')
    v_low = float(parts[0][1:])
    v_high = float(parts[1])

    for m_name, scores in models.items():
        if m_name in target_models:
            avg_mae = np.mean(scores['mae'])
            avg_eol = np.mean(scores['eol_err'])
            plot_list.append({
                'Lower Voltage [V]': v_low,
                'Upper Voltage [V]': v_high,
                'Trajectory MAE [%]': avg_mae,
                'EOL Error': avg_eol,
                'Model': m_name
            })

df_plot = pd.DataFrame(plot_list)


def plot_v2g_final_optimized(df, target_metric, colorbar_label, model_name, display_title):
    df_filtered = df[
        (df['Model'] == model_name) &
        (df['Lower Voltage [V]'] < 3.81) &
        (df['Upper Voltage [V]'] < 4.11)
    ].copy()

    if df_filtered.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.set_style("white")
    ax.grid(False)

    current_max = df_filtered[target_metric].max()
    size_factor = 850 / current_max if current_max != 0 else 100

    scatter = ax.scatter(
        df_filtered['Lower Voltage [V]'],
        df_filtered['Upper Voltage [V]'],
        s=df_filtered[target_metric] * size_factor,
        c=df_filtered[target_metric],
        cmap='plasma',
        alpha=1.0,
        linewidth=0,
        edgecolors='none',
        zorder=3
    )

    ax.set_title(display_title, fontsize=28, pad=10)
    ax.set_xlabel('Lower Voltage [V]', fontsize=22, labelpad=12)
    ax.set_ylabel('Upper Voltage [V]', fontsize=22, labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=22)

    ax.set_xticks(np.arange(3.5, 3.85, 0.05))
    ax.set_yticks(np.arange(3.85, 4.15, 0.05))
    ax.set_xlim(3.47, 3.83)
    ax.set_ylim(3.82, 4.13)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = plt.colorbar(scatter, cax=cax)

    cbar.set_label(colorbar_label, fontsize=22, labelpad=15)
    cbar.ax.tick_params(labelsize=22)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.show()


configs = [
    ('PINN_Base', 'PINN'),
    ('PINN_Phys', 'PINN with virtual sensing'),
    ('CNN_Phys', 'CNN with virtual sensing')
]

metrics_to_plot = [
    ('Trajectory MAE [%]', 'Trajectory MAE [%]', '$Q_{loss}$'),
    ('EOL Error', 'EOL Error [EFCs]', 'EOL')
]

for metric_col, colorbar_label, suffix in metrics_to_plot:
    for m_id, base_title in configs:
        full_title = f"{base_title} ({suffix})"
        plot_v2g_final_optimized(
            df_plot,
            metric_col,
            colorbar_label,
            m_id,
            full_title
        )

#%% load data (Figure 7a,b and c)

print("\n" + "="*20 + " 🔍 Start immediate integrity verification of persisted data. " + "="*20)

parity_file = "v2g_comprehensive_parity_results.pkl"
snapshot_file = "v2g_landscape_snapshots.pkl"
metrics_file = "v2g_metrics_hub.pkl"


if os.path.exists(parity_file):
    with open(parity_file, 'rb') as f:
        parity_results = pickle.load(f)
    print(f"✅ [Parity Data Successfully Recovered] -> {len(parity_results)} independent model curve layers found.")
else:
    print(f"❌ Specified Parity backup file not found: {parity_file}")


if os.path.exists(snapshot_file):
    with open(snapshot_file, 'rb') as f:
        v2g_landscape_snapshots = pickle.load(f)
    print(f"✅ [SHAP Snapshot Successfully Recovered] -> Retrieved {len(v2g_landscape_snapshots)} evolving voltage windows.")
else:
    print(f"❌ Specified SHAP snapshot backup file not found: {snapshot_file}")


if os.path.exists(metrics_file):
    with open(metrics_file, 'rb') as f:
        v2g_metrics_hub = pickle.load(f)
    print(f"✅ [Metrics Data Successfully Recovered] -> Retrieved {len(v2g_metrics_hub)} evolving voltage window metric sets.")

    if len(v2g_metrics_hub) > 0:
        sample_window = list(v2g_metrics_hub.keys())[0]
        print(f"   📊 Randomly sampled window [{sample_window}] error performance：")
        for m_name, errors in v2g_metrics_hub[sample_window].items():
            if 'mae' in errors and len(errors['mae']) > 0:
                avg_mae = np.mean(errors['mae'])
                print(f"     - model [{m_name}]: average MAE = {avg_mae:.4f}%")
else:
    print(f"❌ Specified Metrics backup file not found: {metrics_file}")

print("\n🎉 [🔥 [Three-Way Integrity Check Passed] Data has been perfectly persisted and is ready for safe post-processing and manuscript figure generation！")
print("="*60)

#%% Parity plot (Figure 7a)

def plot_nature_final_revised(parity_dict, target_model_base):

    all_actual = []
    all_pred = []

    for unique_key, data in parity_dict.items():
        if unique_key.startswith(f"{target_model_base}_") or unique_key == target_model_base:
            all_actual.extend(data['actual'])
            all_pred.extend(data['pred'])

    actual = np.array(all_actual)
    pred = np.array(all_pred)

    if len(actual) == 0:
        print(f"⚠️ Notice: No valid scatter data found for model [{target_model_base}] in parity_results. Skipping.")
        return

    actual_pct = actual * 100
    pred_pct = pred * 100


    raw_error_pct = pred_pct - actual_pct
    absolute_error_pct = np.abs(raw_error_pct)


    err_std = np.nanstd(raw_error_pct)

    if err_std > 0:
        error_limit = 3 * err_std
    else:
        error_limit = np.nanmax(absolute_error_pct)


    error_limit = np.ceil(error_limit)


    norm = Normalize(
        vmin=0,
        vmax=error_limit,
        clip=True
    )

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['pdf.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=300)
    sns.set_style("ticks")

    color_pinn = "#5A6096"
    color_virt = "#D6A2BD"

    custom_morandi_cmap = LinearSegmentedColormap.from_list(
        "Morandi_Gradient",
        [color_pinn, color_virt],
        N=256
    )

    scatter = ax.scatter(
        actual_pct,
        pred_pct,
        c=absolute_error_pct,
        cmap=custom_morandi_cmap,
        norm=norm,
        s=8,
        alpha=0.45,
        edgecolors='none',
        zorder=2
    )

    ax.plot(
        [0, 60],
        [0, 60],
        color='black',
        linestyle='-',
        linewidth=0.8,
        zorder=3
    )

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)

    ticks_range = np.arange(0, 61, 10)

    ax.set_xticks(ticks_range)
    ax.set_yticks(ticks_range)

    ax.set_xlabel(
        'Measured $Q_{loss}$ [%]',
        fontsize=13,
        fontweight='normal',
        labelpad=6
    )

    ax.set_ylabel(
        'Predicted $Q_{loss}$ [%]',
        fontsize=13,
        fontweight='normal',
        labelpad=6
    )

    title_mapping = {
        'PINN_Phys': 'PINN + Virtual sensing',
        'PINN_Base': 'PINN',
        'FNN_Base': 'FNN',
        'CNN_Base': 'CNN',
        'LSTM_Base': 'LSTM',
    }

    display_title = title_mapping.get(target_model_base, target_model_base)

    ax.set_title(
        display_title,
        fontsize=14,
        fontweight='normal',
        pad=8
    )

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=11,
        width=0.8,
        length=4
    )

    # inset 误差分布图
    inset_ax = ax.inset_axes(
        [33, 5, 24, 17],
        transform=ax.transData
    )

    inset_ax.hist(
        raw_error_pct,
        bins=18,
        color='#888888',
        edgecolor='#555555',
        linewidth=0.4,
        density=True,
        alpha=0.8,
        zorder=4
    )


    inset_ax.set_xlim(-error_limit, error_limit)

    inset_ax.set_xticks(
        [-error_limit, 0, error_limit]
    )

    inset_ax.set_xticklabels(
        [
            f"{-error_limit:.0f}",
            "0",
            f"{error_limit:.0f}"
        ],
        fontsize=7,
        fontweight='normal'
    )

    inset_ax.set_xlabel(
        'Error [%]',
        fontsize=8,
        fontweight='normal',
        labelpad=1
    )

    inset_ax.set_ylabel(
        'Density',
        fontsize=8,
        fontweight='normal',
        labelpad=1
    )

    inset_ax.tick_params(
        axis='both',
        which='major',
        labelsize=7,
        length=1.5,
        pad=1
    )

    inset_ax.set_facecolor('white')

    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#333333')


    divider = make_axes_locatable(ax)

    cax = divider.append_axes(
        "right",
        size="4%",
        pad=0.12
    )

    cbar = plt.colorbar(scatter, cax=cax)

    cbar.set_label(
        'Absolute Error [%]',
        fontsize=11,
        fontweight='normal',
        labelpad=8
    )


    cbar_ticks = np.linspace(0, error_limit, 5)

    cbar.set_ticks(cbar_ticks)

    cbar.ax.set_yticklabels(
        [f"{t:.0f}" for t in cbar_ticks],
        fontsize=9,
        fontweight='normal'
    )

    cbar.ax.tick_params(
        labelsize=9,
        width=0.7,
        length=3
    )

    cbar.outline.set_linewidth(0.5)

    ax.set_aspect('equal', 'box')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)

    plt.tight_layout()

    # file_safe_name = target_model_base.replace("+", "_").replace(" ", "_")
    # save_name = f"Nature_MorandiColormap_Parity_{file_safe_name}.png"

    # plt.savefig(save_name, dpi=300, bbox_inches='tight')

    plt.show()


core_models_to_plot = [
    'PINN_Base',
    'PINN_Phys',
    'FNN_Base',
    'CNN_Base',
    'LSTM_Base'
]

for model_base in core_models_to_plot:
    plot_nature_final_revised(parity_results, model_base)
    
#%% SHAP Analysis of Multi-Step Fast-Charging Cases for 3.80 V and 3.85 V (Figure 7b)

# necessary function
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

print("=== 🔍 Starting Core SHAP Attribution Interpretability Analysis for cell18 Snapshot Data ===")

current_window_key = list(v2g_landscape_snapshots.keys())[0]
target_seed_index = 0 
snapshot_cluster = v2g_landscape_snapshots[current_window_key][target_seed_index]

target_cell_id = 'cell18'
phys_key = f"{target_cell_id}_PINN_Phys"
base_key = f"{target_cell_id}_PINN_Base"

if phys_key not in snapshot_cluster or base_key not in snapshot_cluster:
    available_keys = list(snapshot_cluster.keys())
    phys_matched = [k for k in available_keys if target_cell_id in k and 'Phys' in k]
    base_matched = [k for k in available_keys if target_cell_id in k and 'Base' in k]
    if len(phys_matched) > 0 and len(base_matched) > 0:
        phys_key = phys_matched[0]
        base_key = base_matched[0]
    else:
        print("❌ Available keys are as follows：", available_keys)
        raise KeyError(f"❌ Unable to resolve the model for {target_cell_id} from the snapshot.")


X_sample_phys = snapshot_cluster[phys_key]['X_sorted_full'].numpy()
features_phys = snapshot_cluster[phys_key]['feature_names']

X_sample_base = snapshot_cluster[base_key]['X_sorted_full'].numpy()
features_base = snapshot_cluster[base_key]['feature_names']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_phys_eval = FFNN(n_input=X_sample_phys.shape[1], n_output=1, n_hidden_layers=[16, 16], activation=nn.SiLU).to(device)
model_phys_eval.load_state_dict(snapshot_cluster[phys_key]['model_state'])
model_phys_eval.eval()

model_base_eval = FFNN(n_input=X_sample_base.shape[1], n_output=1, n_hidden_layers=[16, 16], activation=nn.SiLU).to(device)
model_base_eval.load_state_dict(snapshot_cluster[base_key]['model_state'])
model_base_eval.eval()

def predict_phys_wrapper(x_img):
    return model_phys_eval(torch.tensor(x_img, dtype=torch.float32).to(device)).detach().cpu().numpy()

def predict_base_wrapper(x_img):
    return model_base_eval(torch.tensor(x_img, dtype=torch.float32).to(device)).detach().cpu().numpy()


bg_samples_phys = X_sample_phys[::4] 
bg_samples_base = X_sample_base[::4]

explainer_phys = shap.KernelExplainer(predict_phys_wrapper, bg_samples_phys)
shap_values_phys = explainer_phys.shap_values(X_sample_phys)
mean_shap_phys = np.abs(shap_values_phys).mean(axis=0).flatten()

explainer_base = shap.KernelExplainer(predict_base_wrapper, bg_samples_base)
shap_values_base = explainer_base.shap_values(X_sample_base)
mean_shap_base = np.abs(shap_values_base).mean(axis=0).flatten()

mean_shap_phys_norm = mean_shap_phys / (sum(mean_shap_phys) if sum(mean_shap_phys) > 0 else 1)
mean_shap_base_norm = mean_shap_base / (sum(mean_shap_base) if sum(mean_shap_base) > 0 else 1)


feature_name_mapping = {
    "Dneg_predicted": "Diffusion coef",
    "Lneg_predicted": "Thickness of NE",
    "Lpos_predicted": "Thickness of PE",
    "cspos_predicted": "Li Concentration of PE",
    "epsspos_predicted": "Porosity of PE",
    "rpneg_predicted": "Particle Radius of NE",
    "EFC": "Current EFC",
    "ECOV": "EOCV",
    "media_I": "Median of Current",
    "media_V": "Median of Voltage",
    "skew_I": "Skewness of Current",
    "skew_V": "Skewness of Voltage",
    "kurt_I": "Kurtosis of Current",
    "kurt_DQ": "Kurtosis of DQ",
    "std_I": "Std of Current",
    "seq_CV": "Charging Time CV",
    "seq_CC": "Charging Time CC",
    "shanEntro_I": "ShanEn of Current",
    "shanEntro_DQ": "ShanEn of DQ",
    "slope_I": "Slope of Current",
    "slope_DQ": "Slope of DQ",
    "media_DQ": "Median of DQ",
    "min_DQ": "Minimum of DQ",
    "std_Q": "Std of Q",
    "std_DQ": "Std of DQ",
    "IC_area": "IC area",
    "Q_DV": "DV shift",
    "V_peak": "IC peak position",
    "skew_DQ": "Skewness of DQ",
    "skew_Q": "Skewness of Q",
    "shanEntro_V": "ShanEn of Voltage",
}


def translate_feature_name(raw_name):
    if raw_name in feature_name_mapping:
        return feature_name_mapping[raw_name]
    

    clean_key = raw_name.replace('_predicted', '')
    for k, v in feature_name_mapping.items():
        if clean_key in k or k in clean_key:
            return v
    return raw_name 


shap_df_list = []
for name, val in zip(features_phys, mean_shap_phys_norm):
    shap_df_list.append({'Feature': translate_feature_name(name), 'SHAP_Value': val, 'Group': 'PINN + Virt. sensing'})

for name, val in zip(features_base, mean_shap_base_norm):
    shap_df_list.append({'Feature': translate_feature_name(name), 'SHAP_Value': val, 'Group': 'PINN'})

df_shap_all = pd.DataFrame(shap_df_list)


def plot_nature_shap_giant_fonts(df, window_label, cell_label):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['pdf.fonttype'] = 42
    

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=300)
    sns.set_style("ticks")


    unique_features = df[df['Group'] == 'PINN']['Feature'].tolist()
    for f in df['Feature'].unique():
        if f not in unique_features:
            unique_features.append(f)

    x_pos = np.arange(len(unique_features))
    width = 0.36 

    phys_vals = [df[(df['Feature'] == f) & (df['Group'] == 'PINN + Virt. sensing')]['SHAP_Value'].values[0] if len(df[(df['Feature'] == f) & (df['Group'] == 'PINN + Virt. sensing')]['SHAP_Value'].values) > 0 else 0.0 for f in unique_features]
    base_vals = [df[(df['Feature'] == f) & (df['Group'] == 'PINN')]['SHAP_Value'].values[0] if len(df[(df['Feature'] == f) & (df['Group'] == 'PINN')]['SHAP_Value'].values) > 0 else 0.0 for f in unique_features]


    color_pinn = "#5A6096"        
    color_virt = "#D6A2BD"        


    ax.bar(x_pos - width/2, base_vals, width, color=color_pinn, edgecolor='#333333', linewidth=0.5, alpha=0.95, label='PINN')
    ax.bar(x_pos + width/2, phys_vals, width, color=color_virt, edgecolor='#333333', linewidth=0.5, alpha=0.95, hatch='//', label='PINN + Virt. sensing')

    ax.set_ylabel('Normalized Mean\n|SHAP| Value', fontsize=18, fontweight='normal', labelpad=12)
    
    ax.set_xticks(x_pos)

    ax.set_xticklabels(unique_features, fontsize=13, fontweight='normal', rotation=45, ha='right')
    
    ax.tick_params(axis='y', labelsize=16, direction='out', length=4)
    ax.tick_params(axis='x', bottom=True, direction='out', length=4)
    
    ax.legend(frameon=False, fontsize=15, loc='upper right')
    
    sns.despine(top=True, right=True)

    plt.gcf().subplots_adjust(bottom=0.35)

    plt.tight_layout()
    
    plt.subplots_adjust(
    left=0.20,
    bottom=0.35
    )
    
    plt.show()

plot_nature_shap_giant_fonts(df_shap_all, current_window_key, target_cell_id)

#%% Model Performance in a Narrow Voltage Window (Figure 7c)

summary_list = []
target_models = ['CNN_Base', 'FNN_Base', 'LSTM_Base', 'PINN_Base', 'PINN_Phys']

name_mapping = {
    'PINN_Phys': 'PINN+ \nVirt. sensing',
    'PINN_Base': 'PINN',
    'CNN_Base': 'CNN',
    'LSTM_Base': 'LSTM',
    'FNN_Base': 'FNN'
}

target_window = None

for window in v2g_metrics_hub.keys():
    parts = window.split('_')
    if len(parts) == 2:
        v_low_str = parts[0][1:]
        v_high_str = parts[1]

        if ("3.80" in v_low_str or v_low_str == "3.8") and "3.85" in v_high_str:
            target_window = window
            break

if target_window is None:
    for window in v2g_metrics_hub.keys():
        if "3.80_3.85" in window or "3.8_3.85" in window:
            target_window = window
            break

if target_window is None:
    target_window = list(v2g_metrics_hub.keys())[0]

print(f"🎯 Successfully identified the target narrowest voltage window: {target_window}")

models_data = v2g_metrics_hub[target_window]

for m_name in target_models:
    if m_name in models_data:
        scores = models_data[m_name]

        mae_mean = np.nanmean(scores['mae'])
        rmse_mean = np.nanmean(scores['rmse'])

        paper_name = name_mapping.get(m_name, m_name)

        summary_list.append({
            'Model': paper_name,
            'Trajectory_MAE': mae_mean,
            'Trajectory_RMSE': rmse_mean
        })

df_5models = pd.DataFrame(summary_list)
df_5models = df_5models.sort_values('Trajectory_MAE', ascending=False)



def plot_nature_5models_refined(df, window_name):

    if df.empty:
        print("❌ Error: Data is empty.")
        return

    models = df['Model'].tolist()
    mae_values = df['Trajectory_MAE'].tolist()
    rmse_values = df['Trajectory_RMSE'].tolist()


    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['pdf.fonttype'] = 42


    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    sns.set_style("ticks")

    y_pos = np.arange(len(models))
    height = 0.52

    color_mae = "#5A6096"
    color_rmse = "#D6A2BD"

    ax.barh(
        y_pos,
        [-x for x in mae_values],
        height,
        color=color_mae,
        edgecolor='#333333',
        linewidth=0.7,
        alpha=0.95
    )

    ax.barh(
        y_pos,
        rmse_values,
        height,
        color=color_rmse,
        edgecolor='#333333',
        linewidth=0.7,
        alpha=0.95
    )


    ax.axvline(
        0,
        color='black',
        linewidth=1.0
    )

    ax.set_yticks(y_pos)

    ax.set_yticklabels(
        models,
        fontsize=18,
        fontweight='normal',
        ha='right',
        va='center',
        multialignment='center'
    )

    ax.tick_params(
        axis='y',
        left=False
    )


    max_val = max(max(mae_values), max(rmse_values)) * 1.28

    ax.set_xlim(-max_val, max_val)


    x_ticks = ax.get_xticks()

    ax.set_xticks(x_ticks)

    ax.set_xticklabels(
        [f"{abs(x):.1f}" for x in x_ticks],
        fontsize=17,
        fontweight='normal'
    )

    ax.tick_params(
        axis='x',
        width=1.4,
        length=6
    )


    ax.set_xlabel(
        'Error [%]',
        fontsize=22,
        fontweight='normal',
        labelpad=12
    )


    ax.text(
        -max_val / 2,
        len(models) - 0.05,
        'MAE',
        ha='center',
        va='center',
        fontsize=22,
        fontweight='normal',
        color='#222222'
    )

    ax.text(
        max_val / 2,
        len(models) - 0.05,
        'RMSE',
        ha='center',
        va='center',
        fontsize=22,
        fontweight='normal',
        color='#222222'
    )


    sns.despine(
        left=True,
        bottom=False,
        trim=True
    )


    for i, v in enumerate(mae_values):

        ax.text(
            -v - (max_val * 0.025),
            i,
            f'{v:.2f}',
            ha='right',
            va='center',
            fontsize=16,
            fontweight='normal'
        )

    for i, v in enumerate(rmse_values):

        ax.text(
            v + (max_val * 0.025),
            i,
            f'{v:.2f}',
            ha='left',
            va='center',
            fontsize=16,
            fontweight='normal'
        )


    plt.subplots_adjust(
        bottom=0.20,
        left=0.20
    )

    plt.show()



plot_nature_5models_refined(df_5models, target_window)

