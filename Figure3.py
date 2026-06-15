# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:08:37 2026

Figure 3 replica

@author: wengu476
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_process_functionV2 import process_and_analyze_battery_data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

#%% Embed physical parameters into randomly selected training and test datasets. This function must be run before processing the training and test data without a validation set.


def load_cells_to_dict(folder, cell_list):
    all_data = {}
    for cell_id in cell_list:
        file_path = os.path.join(folder, f"{cell_id}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                all_data[cell_id] = pickle.load(f)
            print(f"✅ Loaded {cell_id} successfully.")
        else:
            print(f"❌ Error: {file_path} not found.")
    return all_data


data_folder = 'data' 

cell_ids = [
    'cell01', 'cell02', 'cell01', 'cell03', 
    'cell04', 'cell05', 'cell01', 'cell08', 
    'cell06', 'cell07', 'cell09', 'cell10', 
    'cell11', 'cell12', 'cell13', 'cell14', 
    'cell15', 'cell16', 'cell17', 'cell18'
]

data = load_cells_to_dict(data_folder, cell_ids)

#%%

result_epss = process_and_analyze_battery_data('data generation/CCCV generationv1.txt', 'epsspos', 2.9, 4.2)
result_rpneg = process_and_analyze_battery_data('data generation/CCCV generationv2.txt', 'rpneg', 2.9, 4.2)
result_Lneg = process_and_analyze_battery_data('data generation/CCCV generationv3.txt', 'Lneg', 2.9, 4.2)
result_cspos = process_and_analyze_battery_data('data generation/CCCV generationv4.txt', 'cspos', 2.9, 4.2)
result_Lpos = process_and_analyze_battery_data('data generation/CCCV generationv5.txt', 'Lpos', 2.9, 4.2)
result_Dneg = process_and_analyze_battery_data('data generation/CCCV generationv6.txt', 'Dneg', 2.9, 4.2)

#%%

def integrate_virtual_sensing_to_dict(cell_data_dict, result_dicts, var_names):

    print(f"Starting physical feature integration for {len(cell_data_dict)} batteries...")
    

    for cell_id, df in cell_data_dict.items():
        all_predictions = []
        

        for result, var_name in zip(result_dicts, var_names):
 
            predictions = predict_and_plot(
                result, 
                df, 
                var_name, 
                color_idx=0, 
                save_path=None
            )
            if predictions is not None:
                all_predictions.append(predictions)
        

        if len(all_predictions) == len(var_names):
            for pred, name in zip(all_predictions, var_names):
                df[name + '_predicted'] = pred

        else:
            print(f"⚠️ {cell_id}: Mapping failed due to a prediction length mismatch.")
            
    return cell_data_dict

def predict_and_plot(result_dict, feature_df, variable_name, color_idx, save_path=None):

    features_in_model = result_dict['features_df'].columns.drop([variable_name, 'log_'+variable_name])
    
    matching_features = [feature for feature in features_in_model if feature in feature_df.columns]
    

    if matching_features:
        X_feature = feature_df[matching_features]
        scaler = StandardScaler()
        X_feature_scaled = scaler.fit_transform(X_feature) 
    
        log_predictions = result_dict['rf_model'].predict(X_feature_scaled)
        predictions = np.exp(log_predictions)      
        return predictions
    else:
        print("No matching features found between the trained model and feature1 DataFrame.")   

#%% Interpretable Features and Virtual Sensing Features (Figure 3)

feature_full_names = {
    'skew_I': 'Current Skewness',
    'seq_CV': 'Charge Time',
    'kurt_I': 'Current Kurtosis',
    'EOCV': 'EOCV',
    'slope_DQ': 'ΔQ Slope',
    'skew_V': 'Voltage Skewness',
    'shanEntro_V': 'Voltage ShanEn',
    'slope_V': 'Voltage Slope',
    'std_DQ': 'ΔQ Std',
    'V_peak': '1st IC Peak Voltage',
    'media_V': 'Median Voltage',
    'std_V': 'Voltage Std',
    'min_DQ': 'ΔQvar',
    'media_DQ': 'Median ΔQ',
    'shanEntro_I': 'Current ShanEn',
    'slope_I': 'Current Slope',
    'Q_DV': 'DV Shift',
    'std_Q': 'Capacity Std',
    'kurt_DQ': 'ΔQ Kurtosis',
}
phys_variables = ['Dneg', 'Lneg', 'rpneg', 'cspos']
phys_results = [result_Dneg, result_Lneg, result_rpneg, result_cspos]


for cell_id in data.keys():
    columns_to_drop = [c for c in data[cell_id].columns if c.endswith('_predicted')]
    if columns_to_drop:
        data[cell_id].drop(columns=columns_to_drop, inplace=True)

data_phy = data.copy()

for res, var_name in zip(phys_results, phys_variables):
    data_phy = integrate_virtual_sensing_to_dict(
        data_phy,
        [res],
        [var_name]
    )

all_cells_list = [df for df in data_phy.values()]
combined_df = pd.concat(all_cells_list, axis=0, ignore_index=True)

feature_cols = combined_df.columns[:29]
soh_col = combined_df.columns[-5]

correlations = combined_df[feature_cols].corrwith(combined_df[soh_col]).abs()
shared_features = correlations.nlargest(16).index.tolist()

cell_ids = sorted(list(data_phy.keys()))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.7
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

fig, axes = plt.subplots(
    4, 5,
    figsize=(27.5, 20),
    sharey=False
)


colors_warm = sns.color_palette("coolwarm", len(cell_ids))

morandi_refined = ['#5A6096', '#9B829C', '#D6A2BD', '#EBB9CB']
morandi_cmap = LinearSegmentedColormap.from_list(
    "morandi_custom",
    morandi_refined
)
colors_morandi = [
    morandi_cmap(i)
    for i in np.linspace(0, 1, len(cell_ids))
]

scaler = MinMaxScaler()


for row in range(4):
    for col in range(5):
        ax = axes[row, col]

        if col < 4:
            idx = row * 4 + col

            if idx >= len(shared_features):
                ax.axis("off")
                continue

            f_name = shared_features[idx]

            for c_idx, cell_id in enumerate(cell_ids):
                df_cell = data_phy[cell_id]

                soh_col_name = 'SOH' if 'SOH' in df_cell.columns else df_cell.columns[-5]

                x_data = 1.0 - df_cell[soh_col_name].values
                y_data = df_cell[f_name].values

                y_norm = scaler.fit_transform(
                    y_data.reshape(-1, 1)
                ).flatten()

                ax.scatter(
                    x_data,
                    y_norm,
                    color=colors_warm[c_idx],
                    alpha=0.65,
                    s=36,
                    edgecolors='none',
                    label=cell_id
                )

            full_feature_name = feature_full_names.get(f_name, f_name)

            ax.set_title(
                f"Feature {idx + 1}",
                fontsize=24,
                fontweight='normal',
                pad=16
            )

            ax.set_ylabel(
                full_feature_name,
                fontsize=22,
                labelpad=10
            )

            ax.set_xlim(-0.02, 0.65)


        else:
            var_label = phys_variables[row]
            target_phys_col = var_label + '_predicted'

            all_y_phys_raw = []
            cell_phys_raw_list = []

            for cell_id in cell_ids:
                df_cell = data_phy[cell_id]

                soh_col_name = 'SOH' if 'SOH' in df_cell.columns else df_cell.columns[-5]

                x_val = 1.0 - df_cell[soh_col_name].values
                y_val_raw = df_cell[target_phys_col].values

                all_y_phys_raw.extend(y_val_raw)
                cell_phys_raw_list.append((x_val, y_val_raw))

            phys_scaler = MinMaxScaler()
            phys_scaler.fit(
                np.array(all_y_phys_raw).reshape(-1, 1)
            )

            for c_idx, (x_data, y_raw) in enumerate(cell_phys_raw_list):
                y_norm = phys_scaler.transform(
                    y_raw.reshape(-1, 1)
                ).flatten()

                ax.scatter(
                    x_data,
                    y_norm,
                    color=colors_morandi[c_idx],
                    alpha=0.65,
                    s=36,
                    edgecolors='none'
                )

            if 'neg' in var_label:
                y_title = f"${var_label[:-3]}_{{neg}}$"
            elif 'pos' in var_label:
                y_title = f"${var_label[:-3]}_{{pos}}$"
            else:
                y_title = var_label

            ax.set_ylabel(
                y_title,
                fontsize=24,
                labelpad=10
            )

            ax.set_xlim(-0.02, 0.65)

        ax.set_xlabel(
            "Qloss",
            fontsize=22,
            labelpad=10
        )

        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)

        ax.tick_params(
            axis='both',
            which='major',
            direction='out',
            labelsize=20,
            width=1.6,
            length=7
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.7)


handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    [f"Battery {i + 1}" for i in range(len(cell_ids))],
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),
    ncol=6,
    frameon=True,         
    fancybox=False,       
    edgecolor='black',
    fontsize=21,
    title=None,           
    markerscale=1.8,
    columnspacing=2.1,
    handletextpad=0.7,
    borderpad=0.8
)

plt.subplots_adjust(
    bottom=0.15,
    hspace=0.62,
    wspace=0.45
)


plt.show()


