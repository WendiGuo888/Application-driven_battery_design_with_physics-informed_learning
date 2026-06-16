# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:30:40 2026

Figure 6 replica

@author: wengu476
"""

import os
import pickle
import torch
import shap
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter  
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator, MaxNLocator

#%% load source data file

class YourModelClass(torch.nn.Module):
    def __init__(self, input_dim=16): 
        super().__init__()
             
        self.network = torch.nn.Sequential(
            # network.0
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, 16),  # network.0.0.weight & bias
                torch.nn.ReLU()
            ),
            # network.1
            torch.nn.Sequential(
                torch.nn.Linear(16, 16),         # network.1.0.weight & bias
                torch.nn.ReLU()
            ),
            # network.2
            torch.nn.Linear(16, 1)               # network.2.weight & bias 
        )

    def forward(self, x):
        return self.network(x)


def load_all_experiment_results_v2(save_dir, sensing_tag=""):

    suffix_tag = f"_{sensing_tag}" if sensing_tag else ""
    
    print("====================================================")
    print(f"📦 Start full scientific asset loading: {save_dir} (Tag: {sensing_tag if sensing_tag else 'None'})")
    print("====================================================")
    
    results = {}
    
    try:

        landscape_path = os.path.join(save_dir, f"landscape_snapshots{suffix_tag}.pt")
        if os.path.exists(landscape_path):
            results['landscape_snapshots'] = torch.load(landscape_path, weights_only=False)
            print("✓ Loaded successfully: landscape_snapshots")
        else:
            print(f"⚠ Landscape file not found: {landscape_path}")

        parity_path = os.path.join(save_dir, f"v2g_parity_results_accumulated{suffix_tag}.pkl")
        if os.path.exists(parity_path):
            with open(parity_path, "rb") as f:
                results['parity_data'] = pickle.load(f)
            print("✓ Loaded successfully: parity_data")

        # 3. 核心指标字典文件读取
        metrics_files = {
            'seed_eol_absolute': f"seed_eol_absolute{suffix_tag}.pkl",
            'seed_results': f"seed_results{suffix_tag}.pkl",
            'all_metrics': f"all_metrics{suffix_tag}.pkl"
        }
        for key, file_name in metrics_files.items():
            path = os.path.join(save_dir, file_name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    results[key] = pickle.load(f)
                print(f"✓ Loaded successfully: {key}")
            else:
                print(f"⚠ Metrics file not found: {path}")

        df_path = os.path.join(save_dir, f"df_results{suffix_tag}.pkl")
        if os.path.exists(df_path):
            results['df_results'] = pd.read_pickle(df_path)
            print("✓ Loaded successfully: df_results")
            
        print("🎉[Assets Loaded Successfully] All available data loaded.！")
        return results

    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None


if __name__ == "__main__":

    save_dir_final = "Saved_Results_2026" 
    current_tag = "diffusion" 


    data_assets = load_all_experiment_results_v2(save_dir_final, sensing_tag=current_tag)


    if data_assets:
        landscape_snapshots = data_assets.get('landscape_snapshots')
        parity_data = data_assets.get('parity_data')
        df_results = data_assets.get('df_results')
        
        seed_results = data_assets.get('seed_results')     
        seed_eol_absolute = data_assets.get('seed_eol_absolute')  
        all_metrics = data_assets.get('all_metrics')         

        print("\n✅ Full asset deserialization completed successfully！")
        

        if landscape_snapshots is not None:
            try:
                snapshot_key = 'cell18_PINN_Phys'
                cell18_data = None
                found_group_key = None
                

                if isinstance(landscape_snapshots, dict):
                    for group_key, sub_dict in landscape_snapshots.items():
                        if isinstance(sub_dict, dict) and snapshot_key in sub_dict:
                            cell18_data = sub_dict[snapshot_key]
                            found_group_key = group_key
                            break  # 
                

                elif isinstance(landscape_snapshots, list):
                    for idx, sub_snapshot in enumerate(landscape_snapshots):
                        if sub_snapshot and snapshot_key in sub_snapshot:
                            cell18_data = sub_snapshot[snapshot_key]
                            found_group_key = idx
                            break

 
                if cell18_data is not None:
                    print(f"🎯 [Structure Match Successful] '{snapshot_key}' was automatically located in the snapshot key group [{found_group_key}].")

                    X_sorted = cell18_data.get('X_sorted_full')
                    target_prepared = cell18_data.get('target_prepared')

                    model_state = cell18_data.get('model_state')
                    
                    if model_state is not None and X_sorted is not None:
                        

                        base_model = YourModelClass(input_dim=X_sorted.shape[1]) 
                        

                        base_model.load_state_dict(model_state)
                        

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model_tgt = base_model.to(device)
                        model_tgt.eval()  #
                        
             
                        with torch.no_grad():
                            test_output = model_tgt(X_sorted.to(device))
                            
                        print("🎉 [Deep Validation Successful] The cell18 neural network model and weights have been loaded, and the Eval forward-pass verification passed successfully！")
                        print(f"📊 [Verification] Test Output Shape: {test_output.shape}")

                        
                    else:
                        print("⚠ Snapshot data found, but the internal 'model_state' or 'X_sorted_full' is None.")
                else:
                    print(f"❌ Searched all snapshot assets but could not find the '{snapshot_key}' key in any nested dictionary.")

            except Exception as e_unpack:
                print(f"❌ Unpacking or forward-pass validation failed: {e_unpack}")
    else:
        print("❌ Unable to reproduce because the assets were not loaded successfully.")
        
#%% load source data file

class YourModelClass(torch.nn.Module):
    def __init__(self, input_dim=16):  
        super().__init__()
        

        self.network = torch.nn.Sequential(
            # network.0
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, 16),  # network.0.0.weight & bias
                torch.nn.ReLU()
            ),
            # network.1
            torch.nn.Sequential(
                torch.nn.Linear(16, 16),         # network.1.0.weight & bias
                torch.nn.ReLU()
            ),
            # network.2
            torch.nn.Linear(16, 1)               # network.2.weight & bias 
        )

    def forward(self, x):
        return self.network(x)



def load_all_experiment_results_v2(save_dir, sensing_tag=""):

    suffix_tag = f"_{sensing_tag}" if sensing_tag else ""
    
    print("====================================================")
    print(f"📦 Start full scientific asset loading: {save_dir} (Tag: {sensing_tag if sensing_tag else 'None'})")
    print("====================================================")
    
    results = {}
    
    try:

        landscape_path = os.path.join(save_dir, f"landscape_snapshots{suffix_tag}.pt")
        if os.path.exists(landscape_path):
            results['landscape_snapshots'] = torch.load(landscape_path, weights_only=False)
            print("✓ Loaded successfully: landscape_snapshots")
        else:
            print(f"⚠ No landscape file found: {landscape_path}")

        parity_path = os.path.join(save_dir, f"v2g_parity_results_accumulated{suffix_tag}.pkl")
        if os.path.exists(parity_path):
            with open(parity_path, "rb") as f:
                results['parity_data'] = pickle.load(f)
            print("✓ Loaded successfully: parity_data")

        metrics_files = {
            'seed_eol_absolute': f"seed_eol_absolute{suffix_tag}.pkl",
            'seed_results': f"seed_results{suffix_tag}.pkl",
            'all_metrics': f"all_metrics{suffix_tag}.pkl"
        }
        for key, file_name in metrics_files.items():
            path = os.path.join(save_dir, file_name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    results[key] = pickle.load(f)
                print(f"✓ Loaded successfully: {key}")
            else:
                print(f"⚠ No metrics file found: {path}")

        df_path = os.path.join(save_dir, f"df_results{suffix_tag}.pkl")
        if os.path.exists(df_path):
            results['df_results'] = pd.read_pickle(df_path)
            print("✓ Loaded successfully: df_results")

        target_prepared_path = os.path.join(save_dir, f"target_prepared{suffix_tag}.pkl")
        if os.path.exists(target_prepared_path):
            results['target_prepared'] = pd.read_pickle(target_prepared_path)
            print("✓ Loaded successfully: target_prepared")
        else:
            print(f"⚠ No independent target_prepared backup file found: {target_prepared_path}")
            
        print("🎉[Assets Loaded Successfully] All available data loaded successfully!")
        return results

    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None


if __name__ == "__main__":

    save_dir_final = "Saved_Results_2026" 
    current_tag = "diffusion"  


    data_assets = load_all_experiment_results_v2(save_dir_final, sensing_tag=current_tag)


    if data_assets:
        landscape_snapshots = data_assets.get('landscape_snapshots')
        parity_data = data_assets.get('parity_data')
        df_results = data_assets.get('df_results')
        
        seed_results = data_assets.get('seed_results')     
        seed_eol_absolute = data_assets.get('seed_eol_absolute')  
        all_metrics = data_assets.get('all_metrics')         

        print("\n✅ Full asset deserialization completed successfully！")
        

        if landscape_snapshots is not None:
            try:
                snapshot_key = 'cell18_PINN_Phys'
                cell18_data = None
                found_group_key = None
                

                if isinstance(landscape_snapshots, dict):
                    for group_key, sub_dict in landscape_snapshots.items():
                        if isinstance(sub_dict, dict) and snapshot_key in sub_dict:
                            cell18_data = sub_dict[snapshot_key]
                            found_group_key = group_key
                            break  
                

                elif isinstance(landscape_snapshots, list):
                    for idx, sub_snapshot in enumerate(landscape_snapshots):
                        if sub_snapshot and snapshot_key in sub_snapshot:
                            cell18_data = sub_snapshot[snapshot_key]
                            found_group_key = idx
                            break


                if cell18_data is not None:
                    print(f"🎯 [Structure Match Successful] Automatically located '{snapshot_key}' in the snapshot key group [{found_group_key}].")
                    
 
                    X_sorted = cell18_data.get('X_sorted_full')
                    

                    if data_assets.get('target_prepared') is not None:
                        target_prepared = data_assets.get('target_prepared')
                        print("💡 Successfully loaded and mapped the target_prepared variable from the standalone pkl file.")
                    else:
                      
                        target_prepared = cell18_data.get('target_prepared')
                    
            
                    model_state = cell18_data.get('model_state')
                    
                    if model_state is not None and X_sorted is not None:
                        
                        base_model = YourModelClass(input_dim=X_sorted.shape[1]) 
                        
           
                        base_model.load_state_dict(model_state)
                        
            
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model_tgt = base_model.to(device)
                        model_tgt.eval() 
                  
                        with torch.no_grad():
                            test_output = model_tgt(X_sorted.to(device))
                            
                        print("🎉[Deep Validation Successful] The cell18 neural network model and weights have been loaded, and the Eval forward-pass verification passed.")
                        print(f"📊 Test Output Shape: {test_output.shape}")
                        
    
                        
                    else:
                        print("⚠ Snapshot data found, but the internal model_state or X_sorted_full is None.")
                else:
                    print(f"❌ Searched all snapshot assets but could not find the '{snapshot_key}' key in any nested dictionary.")

            except Exception as e_unpack:
                print(f"❌ Unpacking or forward-pass validation failed: {e_unpack}")
    else:
        print("❌ Unable to reproduce because the assets were not loaded successfully.")
        
#%% SHAP Analysis for Unseen Fast-Charging Cases (Figure 6a-1)

def plot_ensemble_shap_heatmap(
    landscape_snapshots,
    target_loss=0.1,
    loss_window=0.03,
    seed=42
):
    import random
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # =========================
    # 1. Morandi colormap
    # =========================
    morandi_refined = [
        '#5A6096',
        '#9B829C',
        '#D6A2BD',
        '#EBB9CB'
    ]

    morandi_cmap = LinearSegmentedColormap.from_list(
        "morandi_shap",
        morandi_refined,
        N=256
    )

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
    }

    all_results = []
    raw_names_last = None
    

    chosen_seed_idx = list(landscape_snapshots.keys())[-1] 
    current_snapshots = landscape_snapshots[chosen_seed_idx]

    print("🚀 Starting aggregated SHAP contribution analysis across battery snapshots...")


    for key, snap in current_snapshots.items():

        if "PINN_Phys" not in key:
            continue

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_state = snap["model_state"]
        X_full = snap["X_sorted_full"].to(device)
        raw_names = snap["feature_names"]
        raw_names_last = raw_names

        model = FFNN(
            n_input=X_full.shape[1],
            n_output=1,
            n_hidden_layers=[16, 16],
            activation=nn.SiLU
        ).to(device)

        model.load_state_dict(model_state)
        model.eval()

        with torch.no_grad():
            preds = model(X_full).detach().cpu().numpy().flatten()

        mask = (
            (preds >= target_loss - loss_window) &
            (preds <= target_loss + loss_window)
        )

        X_real = X_full[mask]

        if len(X_real) < 10:
            top_indices = np.argsort(preds)[-20:]
            X_real = X_full[top_indices]

        bg_idx = np.argsort(preds)[:50]
        bg_data = X_full[bg_idx]

        explainer = shap.GradientExplainer(model, bg_data)

        shap_values = explainer.shap_values(
            X_real,
            nsamples=100
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        all_results.append(mean_abs_shap)

    if len(all_results) == 0:
        print("❌ No results containing PINN_Phys were found.")
        return None


    display_names = [
        feature_name_mapping.get(n, n)
        for n in raw_names_last[:len(all_results[0])]
    ]


    unit_ids = [
        f"unit{i+1:02d}"
        for i in range(len(all_results))
    ]

    shap_matrix = pd.DataFrame(
        all_results,
        index=unit_ids,
        columns=display_names
    )


    mean_importance = shap_matrix.mean(axis=0)
    sorted_columns = mean_importance.sort_values(ascending=False).index
    shap_matrix_sorted = shap_matrix[sorted_columns]


    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"

    n_units, n_features = shap_matrix_sorted.shape


    fig_width = max(11, 0.62 * n_features)
    fig_height = max(2.7, 0.32 * n_units)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        dpi=300
    )

    sns.heatmap(
        shap_matrix_sorted,
        annot=False,
        cmap=morandi_cmap,
        linewidths=0.6,
        linecolor="white",
        square=False,
        cbar_kws={
            "label": "Feature contribution",
            "shrink": 0.82,
            "pad": 0.018,
            "aspect": 18
        },
        ax=ax
    )


    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        shap_matrix_sorted.columns,
        rotation=90,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="normal"
    )

 
    ax.set_yticklabels(
        shap_matrix_sorted.index,
        rotation=0,
        fontsize=13,
        fontweight="normal"
    )


    ax.tick_params(
        axis="x",
        bottom=True,
        top=False,
        length=4,
        width=1,
        pad=6,
        colors="black"
    )

    ax.tick_params(
        axis="y",
        left=False,
        right=False,
        length=0,
        pad=8,
        colors="black"
    )


    for spine in ax.spines.values():
        spine.set_visible(False)


    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=12,
        width=1,
        length=4,
        colors="black"
    )

    cbar.set_label(
        "Feature contribution",
        fontsize=14,
        fontweight="normal",
        rotation=270,
        labelpad=22
    )

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("normal")


    plt.subplots_adjust(
        left=0.08,
        right=0.93,
        top=0.96,
        bottom=0.36
    )

    plt.show()

    return shap_matrix_sorted

shap_df_final = plot_ensemble_shap_heatmap(
    landscape_snapshots,
    target_loss=0.10,
    seed=42
)


#%% SHAP Analysis for Multi-Step Fast-Charging Cases to Assess the Impact of Design Parameters (Figure 6a-2)

def plot_final_optimized_shap_threshold(
    model,
    X_full_tensor,
    target_prepared_df,
    target_loss=0.10,
    loss_window=0.02,
    num_samples=600,
    bg_samples=100,
    shap_nsamples=200,
    seed=2026
):
    import numpy as np
    import torch
    import shap
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import random


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(False) 

    model.eval()
    device = next(model.parameters()).device


    feature_name_mapping = {
        "Dneg_predicted": "Diffusion coef",
        "Lneg_predicted": "Thickness of NE",
        "Lpos_predicted": "Thickness of PE",
        "cspos_predicted": "Li Concentration of PE",
        "epsspos_predicted": "Porosity of PE",
        "rpneg_predicted": "Particle Radius of NE",
        "EFC": "Equivalent Full Cycles",
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
    }

    n_input = X_full_tensor.shape[1]
    raw_feature_names = target_prepared_df.columns[:n_input].tolist()

    display_names = [
        feature_name_mapping.get(name, name)
        for name in raw_feature_names
    ]


    with torch.no_grad():
        preds = model(X_full_tensor.to(device)).cpu().numpy().flatten()

    mask = (preds >= target_loss - loss_window) & (preds <= target_loss + loss_window)
    X_real = X_full_tensor[mask]

    if len(X_real) == 0:
        print("❌ No corresponding degradation data found.")
        return

    print(f"✅ sample: {len(X_real)}")

    X_real = X_real.to(device)


    X_mean = X_real.mean(dim=0, keepdim=True)
    X_std = X_real.std(dim=0, keepdim=True) + 1e-6

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    X_explain = X_mean.repeat(num_samples, 1)

    for i in range(X_explain.shape[1]):
        noise = torch.randn(num_samples, device="cpu", generator=generator).to(device)
        X_explain[:, i] += noise * X_std[0, i] * 1.5


    bg_idx = torch.randperm(len(X_full_tensor), generator=generator)[:bg_samples]
    bg_data = X_full_tensor[bg_idx].to(device)


    import random as _rand
    import numpy as _np
    _rand.seed(seed)
    _np.random.seed(seed)

    explainer = shap.GradientExplainer(model, bg_data)


    shap_values = explainer.shap_values(
        X_explain.to(device),
        nsamples=shap_nsamples
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    X_plot = X_explain.cpu().numpy()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sort_idx = np.argsort(mean_abs)[::-1]

    shap_values = shap_values[:, sort_idx]
    X_plot = X_plot[:, sort_idx]
    display_names = [display_names[i] for i in sort_idx]


    morandi_refined = ['#5A6096', '#9B829C', '#D6A2BD', '#EBB9CB']
    morandi_cmap = LinearSegmentedColormap.from_list("morandi_err", morandi_refined, N=256)


    n_features = len(display_names)
    fig_width = 6.6
    fig_height = max(2.4, 0.20 * n_features)

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(fig_width, fig_height), dpi=300)

    shap.summary_plot(
        shap_values,
        X_plot,
        feature_names=display_names,
        cmap=morandi_cmap,
        show=False,
        plot_size=None,
        alpha=0.75
    )

    ax = plt.gca()
    ax.grid(False)
    ax.axvline(0, color="gray", lw=1.5)

    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlabel("SHAP value (Impact on capacity loss prediction)", fontsize=10)
    ax.set_title("")

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    for col in ax.collections:
        col.set_sizes([12])

    fig = plt.gcf()
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(labelsize=10)
        cbar_ax.set_ylabel("Feature value", fontsize=10, rotation=270, labelpad=14)

    plt.subplots_adjust(left=0.34, right=0.88, bottom=0.22, top=0.93)
    plt.show()
    
plot_final_optimized_shap_threshold(
    model=model_tgt,
    X_full_tensor=X_sorted,
    target_prepared_df=target_prepared,
    target_loss=0.10,  
)

#%% Loss Landscape for Assessing the Impact of Design Parameters (Figure 6b)

class FFNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_layers, activation):
        super().__init__()
        
        layers = []
        
        # Layer 1: input layer → first hidden layer
        layers.append(nn.Sequential(nn.Linear(n_input, n_hidden_layers[0]), activation()))
        
        # Hidden layers: first hidden layer → final hidden layer
        for i in range(1, len(n_hidden_layers)):
            layers.append(nn.Sequential(nn.Linear(n_hidden_layers[i-1], n_hidden_layers[i]), activation()))
        
        # Output
        layers.append(nn.Linear(n_hidden_layers[-1], n_output))
        
        # Combine all layers using nn.Sequential
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
def generate_reproducible_seeds(master_seed=2026, num_seeds=2):
    random.seed(master_seed)
    seeds = random.sample(range(10000, 99999), num_seeds)
    print(f"✅ 10 experimental seeds generated from the master seed {master_seed}: {seeds}")
    return seeds


def get_landscape_data_aligned(model, snap, mode='Phys', range_val=0.4, steps=30, fixed_dirs=None):


    model.load_state_dict(snap['model_state'])
    model.eval()
    
    weights = [p.data.clone() for p in model.parameters()]
    

    if fixed_dirs is None:
        def get_diff_dir(w):
            dirs = [torch.randn_like(p) for p in w]
            norm = torch.sqrt(sum([d.pow(2).sum() for d in dirs]))
            return [d / norm for d in dirs]
        dir_x = get_diff_dir(weights)
        dir_y = get_diff_dir(weights)
    else:
        dir_x, dir_y = fixed_dirs


    coords = np.linspace(-range_val, range_val, steps)
    X, Y = np.meshgrid(coords, coords)
    Z = np.zeros((steps, steps))
    
    X_val, y_val = snap['X_val'], snap['y_val']

    with torch.no_grad():
        for i in range(steps):
            for j in range(steps):
                for p, orig, dx, dy in zip(model.parameters(), weights, dir_x, dir_y):
                    p.data = orig + X[i, j] * dx + Y[i, j] * dy
                
                pred = model(X_val)
                loss = torch.mean((pred - y_val)**2).item()
            
                Z[i, j] = np.log10(loss + 1e-9)


    model.load_state_dict(snap['model_state'])
    return X, Y, Z, (dir_x, dir_y)
    
    
def plot_landscape_comparison(Xb, Yb, Zb, Xp, Yp, Zp):

    plt.rcParams.update(plt.rcParamsDefault) 
    fig = plt.figure(figsize=(26, 11), dpi=150)

    z_min = min(Zb.min(), Zp.min())
    z_max = max(Zb.max(), Zp.max())

    ax1_pos = [0.32, 0.12, 0.23, 0.75]
    cb1_pos = [0.56, 0.25, 0.012, 0.50]
    ax2_pos = [0.67, 0.12, 0.23, 0.75]
    cb2_pos = [0.91, 0.25, 0.012, 0.50]


    ax1 = fig.add_axes(ax1_pos, projection='3d')
    surf1 = ax1.plot_surface(
        Xb, Yb, Zb,
        cmap='terrain',
        vmin=z_min, vmax=z_max,
        edgecolor='none', alpha=0.85, antialiased=True
    )


    ax2 = fig.add_axes(ax2_pos, projection='3d')
    surf2 = ax2.plot_surface(
        Xp, Yp, Zp,
        cmap='plasma',
        vmin=z_min, vmax=z_max,
        edgecolor='none', alpha=0.85, antialiased=True
    )

    axes = [ax1, ax2]
    surfs = [surf1, surf2]
    cb_positions = [cb1_pos, cb2_pos]
    cbar_labels = ["Phys. Loss", "Virt. Loss"]

    for i, (ax, surf, cb_pos, cb_label) in enumerate(zip(axes, surfs, cb_positions, cbar_labels)):
   
        ax.view_init(elev=35, azim=-120)
        ax.set_zlim(z_min, z_max)

       
        ax.set_xlabel(r'Direction $\delta$', fontsize=24, fontweight='normal', labelpad=30)
        ax.set_ylabel(r'Direction $\eta$', fontsize=24, fontweight='normal', labelpad=35)
    
        if ax is ax1:
            ax.set_zlabel('')
        else:
            ax.set_zlabel(r'$\log_{10}(MSE)$', fontsize=24, fontweight='normal', labelpad=25)

        if i == 0:
            ax.zaxis.set_rotate_label(False)
            ax.zaxis.label.set_rotation(90)
            ax.zaxis.set_label_coords(-0.10, 0.5)
        else:
            ax.zaxis.set_rotate_label(False)
            ax.zaxis.label.set_rotation(90)
            ax.zaxis.set_label_coords(-0.06, 0.5)


        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))

        ax.tick_params(axis='both', which='major', labelsize=22, pad=15)


        ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
        ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
        ax.zaxis._axinfo["grid"]['linewidth'] = 0.5

        cax = fig.add_axes(cb_pos)
        cbar = fig.colorbar(surf, cax=cax)

        cbar.set_label(
            rf'{cb_label} ($\log_{{10}}$)',
            fontsize=24, fontweight='normal', labelpad=20
        )
        cbar.ax.tick_params(labelsize=24)

        cb_ticks = np.linspace(z_min, z_max, 4)
        cbar.set_ticks(cb_ticks)
        cbar.ax.set_yticklabels([f"{t:.1f}" for t in cb_ticks])

    fig.text(0.3, 0.52, r'$\log_{10}(MSE)$', rotation=90, fontsize=24, fontweight='normal', va='center', ha='center')

    plt.show()
    


torch.manual_seed(42) #
np.random.seed(42)

target_cell = 'cell14' 

random_seeds = generate_reproducible_seeds(2026, 10)

chosen_seed_idx = len(random_seeds) - 9


snap_b = landscape_snapshots[chosen_seed_idx][f"{target_cell}_PINN_Base"]
snap_p = landscape_snapshots[chosen_seed_idx][f"{target_cell}_PINN_Phys"]


def recreate_model_from_snap(snap, device='cpu'):

    n_in = snap['X_val'].shape[1]

    model = FFNN(n_input=n_in, n_output=1, n_hidden_layers=[16, 16], activation=nn.SiLU).to(device)
    model.load_state_dict(snap['model_state'])
    return model

model_b = recreate_model_from_snap(snap_b, device=device)
model_p = recreate_model_from_snap(snap_p, device=device)


print(f"📊 Mapping the [Base] landscape for {target_cell}...")

Xb, Yb, Zb, directions = get_landscape_data_aligned(model_b, snap_b, mode='Base', steps=40)


print(f"📊 Mapping the [Phys] landscape for {target_cell}...")

Xp, Yp, Zp, _ = get_landscape_data_aligned(model_p, snap_p, mode='Phys', fixed_dirs=directions, steps=40)

plot_landscape_comparison(Xb, Yb, Zb, Xp, Yp, Zp)

#%% Heterogeneity Plot of Virtual Sensing Features (Figure 6c)

def plot_feature_heterogeneity_advanced(
    landscape_snapshots,
    feature_to_plot="Diffusion coef of NE"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    morandi_refined = [
        "#5A6096",
        "#9B829C",
        "#D6A2BD",
        "#EBB9CB"
    ]

    custom_cmap = LinearSegmentedColormap.from_list(
        "hetero_cmap",
        morandi_refined,
        N=256
    )

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"

    # 更接近图1的小尺寸 panel
    fig, ax = plt.subplots(figsize=(3.0, 2.7), dpi=300)

    phys_snaps = {
        k: v for k, v in landscape_snapshots.items()
        if "PINN_Phys" in k
    }

    cell_keys = sorted(phys_snaps.keys())
    n_cells = len(cell_keys)

    print(f"🚀 Plotting the evolution of {feature_to_plot} across {n_cells} batteries...")

    for i, key in enumerate(cell_keys):
        snap = phys_snaps[key]

        X_full = snap["X_sorted_full"].detach().cpu().numpy()
        raw_names = snap["feature_names"]
        M = snap["M"]

        try:
            efc_idx = raw_names.index("EFC")

            if feature_to_plot == "Diffusion coef of NE" and "Dneg_predicted" in raw_names:
                target_idx = raw_names.index("Dneg_predicted")
            else:
                target_idx = raw_names.index(feature_to_plot)

            cycles = X_full[:, efc_idx] * M
            values = X_full[:, target_idx]

            color_val = i / (n_cells - 1) if n_cells > 1 else 0.5

            ax.scatter(
                cycles,
                values,
                color=custom_cmap(color_val),
                s=8,
                alpha=0.65,
                edgecolors="none",
                rasterized=True
            )

        except (ValueError, KeyError):
            continue


    ax.grid(False)

    ax.set_xlabel(
        "Current EFC",
        fontsize=10,
        fontweight="normal",
        labelpad=4
    )

    ax.set_ylabel(
        "Normalized diffusion coef of NE",
        fontsize=10,
        fontweight="normal",
        labelpad=5
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
        width=0.8,
        length=3,
        direction="out"
    )

    for side in ["top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)

    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.8)


    sm = plt.cm.ScalarMappable(
        cmap=custom_cmap,
        norm=plt.Normalize(vmin=0, vmax=10)
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.055,
        fraction=0.055,
        aspect=18
    )

    cbar.set_ticks([0, 10])
    cbar.set_ticklabels(["0", "10"])

    cbar.ax.tick_params(
        labelsize=9,
        width=0.8,
        length=3
    )

    cbar.set_label(
        "Battery unit",
        fontsize=9,
        fontweight="normal",
        rotation=270,
        labelpad=12
    )

    cbar.outline.set_linewidth(0.8)

    plt.tight_layout(pad=0.8)
    plt.show()
    

last_seed = list(landscape_snapshots.keys())[-1]

plot_feature_heterogeneity_advanced(
    landscape_snapshots[last_seed],
    feature_to_plot="Diffusion coef of NE"
)

#%% EOL Prediction Results and Errors Across 10 Random Seeds (Figure 6d)

def plot_eol_comprehensive_analysis_aggregated(seed_eol_absolute):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import r2_score, mean_absolute_error
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.linewidth"] = 1.3

    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 20
    STATS_SIZE = 20

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), dpi=300)

    colors = {
        "PINN_Base": "#5A6096",
        "PINN_Phys": "#D6A2BD"
    }

    m_types = ["PINN_Base", "PINN_Phys"]
    errors_dict = {}


    lim_min = 0
    lim_max = 1000
    x_line = np.linspace(lim_min, lim_max, 300)

    for i, m_type in enumerate(m_types):
        ax = axes[i]

        all_actual = []
        all_pred = []
        all_r2 = []
        all_mae = []

        for s_idx in seed_eol_absolute[m_type].keys():
            act = np.array(seed_eol_absolute[m_type][s_idx]["actual_eol"], dtype=float)
            pre = np.array(seed_eol_absolute[m_type][s_idx]["pred_eol"], dtype=float)

            if len(act) == 0:
                continue

            all_actual.extend(act)
            all_pred.extend(pre)

            if len(act) > 1:
                all_r2.append(r2_score(act, pre))

            all_mae.append(mean_absolute_error(act, pre))

        actual = np.array(all_actual, dtype=float)
        pred = np.array(all_pred, dtype=float)

        if len(actual) == 0:
            print(f"⚠️ No data available for {m_type}. Skipping.")
            continue

        errors_dict[m_type] = pred - actual


        ax.fill_between(
            x_line,
            x_line * 0.90,
            x_line * 1.10,
            color="gray",
            alpha=0.08,
            zorder=0
        )

        ax.plot(
            x_line,
            x_line * 0.90,
            color="gray",
            lw=0.8,
            alpha=0.25,
            zorder=0
        )

        ax.plot(
            x_line,
            x_line * 1.10,
            color="gray",
            lw=0.8,
            alpha=0.25,
            zorder=0
        )

        # Ideal line
        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            linestyle="--",
            color="gray",
            lw=2.2,
            alpha=0.75,
            zorder=1
        )

        ax.scatter(
            actual,
            pred,
            color=colors[m_type],
            s=55,
            alpha=0.38,
            edgecolors="none",
            zorder=2
        )

        mean_r2 = np.mean(all_r2) if len(all_r2) > 0 else np.nan
        std_r2 = np.std(all_r2) if len(all_r2) > 0 else np.nan
        mean_mae = np.mean(all_mae)
        std_mae = np.std(all_mae)

        stats_text = (
            f"Mean R²: {mean_r2:.2f}±{std_r2:.2f}\n"
            f"MAE: {mean_mae:.2f}±{std_mae:.2f}"
        )

        ax.text(
            0.05,
            0.93,
            stats_text,
            transform=ax.transAxes,
            fontsize=STATS_SIZE,
            fontweight="normal",
            va="top",
            ha="left"
        )

        ax.set_xlabel(
            "Observed cycle life [EFCs]",
            fontsize=LABEL_SIZE,
            fontweight="normal",
            labelpad=10
        )

        ax.set_ylabel(
            "Predicted cycle life [EFCs]",
            fontsize=LABEL_SIZE,
            fontweight="normal",
            labelpad=10
        )

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal", adjustable="box")

        ax.tick_params(
            axis="both",
            labelsize=TICK_SIZE,
            width=1.3,
            length=6,
            direction="out"
        )

        ax.grid(
            True,
            alpha=0.15,
            linestyle=":",
            linewidth=0.8
        )

        legend_handles = [
            Line2D(
                [0], [0],
                color="gray",
                linestyle="--",
                lw=2.2,
                label="Ideal"
            ),
            Patch(
                facecolor="gray",
                alpha=0.12,
                edgecolor="none",
                label="±10% Error Band"
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=colors[m_type],
                markeredgecolor="black",
                markersize=8,
                label="Test Batteries"
            )
        ]

        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=LEGEND_SIZE,
            frameon=False,
            handlelength=2.4,
            labelspacing=0.45,
            borderaxespad=0.7
        )

    # Error distribution
    ax_dist = axes[2]

    for m_type in m_types:
        if m_type not in errors_dict:
            continue

        label = "Base" if m_type == "PINN_Base" else "Virt. sensing"

        sns.histplot(
            errors_dict[m_type],
            kde=True,
            ax=ax_dist,
            color=colors[m_type],
            label=label,
            element="step",
            alpha=0.35,
            linewidth=2.2
        )

    ax_dist.axvline(
        0,
        color="black",
        lw=1.5,
        alpha=0.65
    )

    ax_dist.set_xlabel(
        "Error [EFCs]",
        fontsize=LABEL_SIZE,
        fontweight="normal",
        labelpad=10
    )

    ax_dist.set_ylabel(
        "Count",
        fontsize=LABEL_SIZE,
        fontweight="normal",
        labelpad=10
    )

    ax_dist.tick_params(
        axis="both",
        labelsize=TICK_SIZE,
        width=1.3,
        length=6,
        direction="out"
    )

    ax_dist.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
        handlelength=2.2,
        labelspacing=0.45
    )

    ax_dist.grid(
        True,
        alpha=0.15,
        linestyle=":",
        linewidth=0.8
    )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

    plt.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.96,
        bottom=0.18,
        wspace=0.18
    )

    plt.show()


plot_eol_comprehensive_analysis_aggregated(seed_eol_absolute)

#%% computational capability

def plot_academic_summary_bars_final_dual_time(df_results):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    df_plot = df_results.copy()
    df_plot["Traj_R2"] = df_plot["Traj_R2"].clip(lower=-1.0)
    df_plot["EOL_R2"] = df_plot["EOL_R2"].clip(lower=-1.0)

    mode_name_map = {
        "PINN_Base": "Base", "PINN_Phys": "Virt. sensing",
        "Base": "Base", "Phys": "Virt. sensing"
    }
    df_plot["Mode_display"] = df_plot["Mode"].map(mode_name_map).fillna(df_plot["Mode"])

    # 🌟 修复点 1：将 "Time (s)" 替换为新命名 "Train Time (s)"
    time_stats = df_plot.groupby(["Model", "Mode_display"])["Train Time (s)"].mean().unstack()
    model_order = df_plot["Model"].unique()
    time_stats = time_stats.reindex(model_order)

    sns.set_theme(style="white", rc={
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.edgecolor": "black", "axes.linewidth": 2.0
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    metrics_grid = [["Traj_MAE (%)", "EOL_MAE (Cycles)"], ["Traj_R2", "EOL_R2"]]
    y_labels = [[r"Trajectory MAE [%]", r"EOL MAE [EFCs]"], [r"Trajectory R$^2$", r"EOL R$^2$"]]
    custom_palette = {"Base": "#5A6096", "Virt. sensing": "#D6A2BD"}

    legend_handles = [
        Patch(facecolor=custom_palette["Base"], edgecolor="black", alpha=0.30, label="Base"),
        Patch(facecolor=custom_palette["Virt. sensing"], edgecolor="black", alpha=0.30, label="Virt. sensing"),
        Line2D([0], [0], color='#2F3561', marker='o', markersize=6, linestyle='--', linewidth=1.5, label='Time [Base]'),
        Line2D([0], [0], color='#A66384', marker='s', markersize=6, linestyle='-', linewidth=2, label='Time [Virt. sensing]')
    ]

    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            metric = metrics_grid[row][col]
            label = y_labels[row][col]

          
            sns.barplot(
                data=df_plot, x="Model", y=metric, hue="Mode_display",
                hue_order=["Base", "Virt. sensing"], ax=ax, palette=custom_palette,
                errorbar="sd", capsize=0.18, alpha=0.30, edgecolor="black", linewidth=2.0, zorder=1
            )
     
            sns.stripplot(
                data=df_plot, x="Model", y=metric, hue="Mode_display",
                hue_order=["Base", "Virt. sensing"], ax=ax, palette=custom_palette,
                dodge=True, jitter=0.18, size=6, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3
            )

            ax2 = ax.twinx()
            x_indices = np.arange(len(model_order))
            offset = 0.2 

            ax2.plot(x_indices - offset, time_stats["Base"], 
                     color='#2F3561', marker='o', markersize=7, linestyle='--', linewidth=1.5, zorder=5)
            
            ax2.plot(x_indices + offset, time_stats["Virt. sensing"], 
                     color='#A66384', marker='s', markersize=8, linestyle='-', linewidth=2.5, zorder=5)
            
  
            ax2.set_ylim(0, df_plot["Train Time (s)"].max() * 1.6) 
            if col == 1:
                ax2.set_ylabel("Avg. Computation Time [s]", fontsize=18, fontweight='normal', labelpad=15)
            ax2.tick_params(axis='y', labelsize=16, width=2)
            for spine in ax2.spines.values(): spine.set_linewidth(2.0)

   
            if ax.get_legend() is not None: 
                ax.get_legend().remove() #
            
            ax.legend(
                handles=legend_handles, 
                fontsize=14,      
                frameon=True, 
                loc="upper right", 
                ncol=2,             
                columnspacing=0.8
            )

 
            if row == 1:
                ax.axhline(0, color="black", lw=1.5, zorder=2)
                ax.set_ylim(-1.05, 1.4) 
            else:
                ax.set_ylim(0, df_plot[metric].max() * 1.5) 

            ax.set_ylabel(label, fontsize=20, fontweight='normal', labelpad=15)
            ax.set_xlabel("")
            ax.tick_params(axis="both", labelsize=18, width=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, hspace=0.30, wspace=0.38) 
    plt.show()
    
plot_academic_summary_bars_final_dual_time(df_results)

#%% parity plot for virtual sensing impact

morandi_refined = [
    '#5A6096', 
    '#9B829C', 
    '#D6A2BD', 
    '#EBB9CB'  
]
morandi_cmap = LinearSegmentedColormap.from_list("morandi_err", morandi_refined, N=256)

def plot_percentage_parity_morandi(seed_results, m_base, m_phys):

    sns.set_style("white")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    vmax_err = 10.0 
    scatter_objects = []

    axis_min, axis_max = -0.02, 0.62 

    for i, (label, m_name) in enumerate([('PINN', m_base), ('PINN with virtual sensing', m_phys)]):
        ax = axes[i]
        
        all_act = []
        all_pred = []
        for s in seed_results[m_name].keys():
            all_act.extend(seed_results[m_name][s]['actual'])
            all_pred.extend(seed_results[m_name][s]['pred'])
        act_np, pred_np = np.array(all_act), np.array(all_pred)
 
        err_percent = np.abs(act_np - pred_np) * 100
        
        r2 = r2_score(act_np, pred_np)
        mae_percent = np.mean(np.abs(act_np - pred_np)) * 100
        
        sc = ax.scatter(act_np, pred_np, 
                        c=err_percent, 
                        cmap=morandi_cmap,
                        vmin=0, vmax=vmax_err,
                        alpha=0.4, 
                        s=20, edgecolor='none', zorder=2)
        scatter_objects.append(sc)
        
        ax.plot([axis_min, axis_max], [axis_min, axis_max], '--', color='#333333', lw=1.2, zorder=1)
        
        def to_percent(x, position):
            return f"{int(100 * x)}"
        
        ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

        text_str = f"$R^2$: {r2:.3f}\nMAE: {mae_percent:.2f}%"
        ax.text(0.05, 0.92, text_str, transform=ax.transAxes, 
                fontsize=15, fontweight='normal', va='top', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_xlabel('Actual $Q_{loss}$ [%]', fontsize=16, fontweight='normal')
        ax.set_ylabel('Predicted $Q_{loss}$ [%]', fontsize=16, fontweight='normal')
        ax.set_title(label, fontsize=18, fontweight='normal', pad=15)
        
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

    fig.subplots_adjust(right=0.88, wspace=0.3)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(scatter_objects[1], cax=cbar_ax)
    cbar.set_label('Absolute Error [%]', fontsize=14, fontweight='normal')
    
    plt.show()

plot_percentage_parity_morandi(seed_results, 'PINN_Base', 'PINN_Phys')