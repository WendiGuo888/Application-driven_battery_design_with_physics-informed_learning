# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:14:58 2026

Figure 5 replica

@author: wengu476
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

#%% Performance at Different Training Ratios (Figure 5a)

def prepare_plot_df(metrics_dict, scale_name="Training Scale"):
    rows = []
    if not metrics_dict:
        return pd.DataFrame()

    raw_ratios = sorted([float(r) for r in metrics_dict.keys()])

    for ratio in raw_ratios:
        orig_key = next(k for k in metrics_dict.keys() if float(k) == ratio)
        ratio_label = f"{int(ratio * 100)}%"
        models = metrics_dict[orig_key]

        for model_name, values in models.items():
            display_name = "Ours (PINN)" if "PINN" in model_name.upper() else model_name

            for v in values:
                if v is not None and not np.isnan(v):
                    rows.append({
                        scale_name: ratio_label,
                        "Model": display_name,
                        "Error_Value": v
                    })

    return pd.DataFrame(rows)


def plot_mae_eol_violin_flat(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    metrics_to_plot = ["mae", "eol"]
    y_labels = ["Trajectory MAE [%]", "EOL Relative Error [%]"]
    model_order = ["Ours (PINN)", "FNN", "CNN", "LSTM"]


    AXIS_LABEL_SIZE = 15
    TICK_SIZE = 12
    LEGEND_TITLE_SIZE = 11
    LEGEND_SIZE = 10

    colors = ["#F4B5D1", "#F6E0A4", "#A9D4E9", "#B3A2C8"]
    academic_cmap = mcolors.LinearSegmentedColormap.from_list(
        "pinn_custom",
        colors,
        N=256
    )

    sns.set_theme(style="white")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 2.8),
        dpi=300
    )

    for ax, m_key, y_label in zip(axes, metrics_to_plot, y_labels):
        if m_key not in data:
            ax.axis("off")
            continue

        df = prepare_plot_df(data[m_key])

        if df.empty:
            ax.axis("off")
            continue

        ratio_order = sorted(
            df["Training Scale"].unique(),
            key=lambda x: int(x.replace("%", ""))
        )

        n_ratios = len(ratio_order)
        current_order = [m for m in model_order if m in df["Model"].unique()]

        color_indices = np.linspace(0, 1, n_ratios)
        current_palette = [academic_cmap(idx) for idx in color_indices]

        sns.violinplot(
            x="Model",
            y="Error_Value",
            hue="Training Scale",
            data=df,
            order=current_order,
            hue_order=ratio_order,
            palette=current_palette,
            inner=None,
            cut=2,
            alpha=0.65,
            linewidth=0,
            bw_method=0.45,
            ax=ax
        )

        sns.stripplot(
            x="Model",
            y="Error_Value",
            hue="Training Scale",
            data=df,
            order=current_order,
            hue_order=ratio_order,
            dodge=True,
            color="#555555",
            alpha=0.28,
            size=2.2,
            jitter=0.15,
            legend=False,
            ax=ax,
            zorder=2
        )

        # average red line
        for i_m, model in enumerate(current_order):
            for i_r, ratio in enumerate(ratio_order):
                subset = df[
                    (df["Model"] == model) &
                    (df["Training Scale"] == ratio)
                ]["Error_Value"]

                if not subset.empty:
                    mean_val = subset.mean()
                    width = 0.8 / n_ratios
                    offset = (i_r - (n_ratios - 1) / 2) * width
                    center_pos = i_m + offset

                    ax.hlines(
                        mean_val,
                        center_pos - width / 4,
                        center_pos + width / 4,
                        colors="#D62728",
                        lw=2.0,
                        zorder=10
                    )

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

        ax.set_ylabel(
            y_label,
            fontsize=AXIS_LABEL_SIZE,
            fontweight="normal",
            labelpad=8
        )
        ax.set_xlabel("")

        ax.tick_params(axis="x", labelsize=TICK_SIZE, width=1.2, length=4)
        ax.tick_params(axis="y", labelsize=TICK_SIZE, width=1.2, length=4)

        ax.grid(False)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        legend_handles = [
            Patch(
                facecolor=current_palette[i],
                edgecolor="none",
                alpha=0.65,
                label=ratio
            )
            for i, ratio in enumerate(ratio_order)
        ]

        ax.legend(
            handles=legend_handles,
            title="Train Scale",
            title_fontsize=LEGEND_TITLE_SIZE,
            fontsize=LEGEND_SIZE,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.60,
            loc="upper left",
            bbox_to_anchor=(0.00, 1.02),
            borderaxespad=0.0
        )

    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.22,
        top=0.92,
        wspace=0.25
    )

    plt.show()


plot_mae_eol_violin_flat("Results_20260321_1739.pkl")

#%% Performance at Different Test Ratios (Figure 5b)

def prepare_test_plot_df(metrics_dict, scale_name="Test Scale"):
    rows = []
    if not metrics_dict:
        return pd.DataFrame()

    raw_ratios = sorted([float(r) for r in metrics_dict.keys()])

    for ratio in raw_ratios:
        orig_key = next(k for k in metrics_dict.keys() if float(k) == ratio)
        ratio_label = f"{int(ratio * 100)}%"
        models = metrics_dict[orig_key]

        for model_name, values in models.items():
            display_name = "Ours (PINN)" if "PINN" in model_name.upper() else model_name

            for v in values:
                if v is not None and not np.isnan(v):
                    rows.append({
                        scale_name: ratio_label,
                        "Model": display_name,
                        "Error_Value": v
                    })

    return pd.DataFrame(rows)


def plot_test_mae_eol_violin_flat(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    metrics_to_plot = ["mae", "eol"]
    y_labels = ["Trajectory MAE [%]", "EOL Relative Error [%]"]
    model_order = ["Ours (PINN)", "FNN", "CNN", "LSTM"]

    AXIS_LABEL_SIZE = 15
    TICK_SIZE = 12
    LEGEND_TITLE_SIZE = 11
    LEGEND_SIZE = 10

    colors = ["#F4B5D1", "#F6E0A4", "#A9D4E9", "#B3A2C8"]
    academic_cmap = mcolors.LinearSegmentedColormap.from_list(
        "pinn_custom",
        colors,
        N=256
    )

    sns.set_theme(style="white")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 2.8),
        dpi=300
    )

    for ax, m_key, y_label in zip(axes, metrics_to_plot, y_labels):
        if m_key not in data:
            ax.axis("off")
            continue

        df = prepare_test_plot_df(data[m_key], scale_name="Test Scale")

        if df.empty:
            ax.axis("off")
            continue

        ratio_order = sorted(
            df["Test Scale"].unique(),
            key=lambda x: int(x.replace("%", ""))
        )

        n_ratios = len(ratio_order)
        current_order = [m for m in model_order if m in df["Model"].unique()]

        color_indices = np.linspace(0, 1, n_ratios)
        current_palette = [academic_cmap(idx) for idx in color_indices]

        sns.violinplot(
            x="Model",
            y="Error_Value",
            hue="Test Scale",
            data=df,
            order=current_order,
            hue_order=ratio_order,
            palette=current_palette,
            inner=None,
            cut=2,
            alpha=0.65,
            linewidth=0,
            bw_method=0.45,
            ax=ax
        )

        sns.stripplot(
            x="Model",
            y="Error_Value",
            hue="Test Scale",
            data=df,
            order=current_order,
            hue_order=ratio_order,
            dodge=True,
            color="#555555",
            alpha=0.28,
            size=2.2,
            jitter=0.15,
            legend=False,
            ax=ax,
            zorder=2
        )

        for i_m, model in enumerate(current_order):
            for i_r, ratio in enumerate(ratio_order):
                subset = df[
                    (df["Model"] == model) &
                    (df["Test Scale"] == ratio)
                ]["Error_Value"]

                if not subset.empty:
                    mean_val = subset.mean()
                    width = 0.8 / n_ratios
                    offset = (i_r - (n_ratios - 1) / 2) * width
                    center_pos = i_m + offset

                    ax.hlines(
                        mean_val,
                        center_pos - width / 4,
                        center_pos + width / 4,
                        colors="#D62728",
                        lw=2.0,
                        zorder=10
                    )

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

        ax.set_ylabel(
            y_label,
            fontsize=AXIS_LABEL_SIZE,
            fontweight="normal",
            labelpad=8
        )
        ax.set_xlabel("")

        ax.tick_params(axis="x", labelsize=TICK_SIZE, width=1.2, length=4)
        ax.tick_params(axis="y", labelsize=TICK_SIZE, width=1.2, length=4)

        ax.grid(False)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        legend_handles = [
            Patch(
                facecolor=current_palette[i],
                edgecolor="none",
                alpha=0.65,
                label=ratio
            )
            for i, ratio in enumerate(ratio_order)
        ]

        ax.legend(
            handles=legend_handles,
            title="Test Scale",
            title_fontsize=LEGEND_TITLE_SIZE,
            fontsize=LEGEND_SIZE,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.60,
            loc="upper left",
            bbox_to_anchor=(0.00, 1.02),
            borderaxespad=0.0
        )

    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.22,
        top=0.92,
        wspace=0.25
    )


    plt.show()


plot_test_mae_eol_violin_flat("Results_Fixed_20260323_1605_RESCUE.pkl")

#%% Parity plot (Figure 5c)

load_filename = "PINN_trajectories_20260429_1402.pkl"

print("=== 📂 Launch the recovery of full-lifecycle capacity degradation trajectory assets ===")

if os.path.exists(load_filename):
    with open(load_filename, 'rb') as f:

        all_trajectories_tgt = pickle.load(f)
        
    print(f"✅ Data loaded successfully. File path: {os.path.abspath(load_filename)}")
    print("-" * 60)
    

    try:
        # Level 1: Voltage Window
        windows = list(all_trajectories_tgt.keys())
        print(f"📌 Level 1: Evolving Voltage Windows ({len(windows)} in total):")
        for w in windows[:4]: 
            print(f"  - {w}")
        if len(windows) > 4: print("  - ... ")
        

        sample_w = windows[0]
        second_layer = all_trajectories_tgt[sample_w]
        
        if isinstance(second_layer, dict):
            keys_l2 = list(second_layer.keys())
  
            print(f"\n📌 Level 2 Inspection (Voltage Window: [{sample_w}], {len(keys_l2)} Keys):")
            print(f"  - Available Sample Keys: {keys_l2[:5]}")
            
            sample_l2 = keys_l2[0]
            third_layer = second_layer[sample_l2]
            if isinstance(third_layer, dict):
                keys_l3 = list(third_layer.keys())
                print(f"\n📌 Level 3 Sample (Including the Following Data Elements/Model Trajectories):")
                print(f"  - current element: {keys_l3}")
                
    except Exception as err:
        print(f"⚠️ Minor discrepancy detected during automatic structure inspection (data loading is unaffected): {err}")

else:
    print(f"❌ Critical error! File not found in the current working directory: {load_filename}")
    print("💡 Please make sure the .pkl file has been copied to the same directory as your current Jupyter notebook or Spyder script.")
    all_trajectories_tgt = {}

print("=" * 60)


def plot_parity_and_density_minimal(trajectories_dict):

    morandi_refined = ["#5A6096", "#9B829C", "#D6A2BD", "#EBB9CB"]

    point_cmap = LinearSegmentedColormap.from_list(
        "morandi_points",
        morandi_refined,
        N=256
    )

    contour_cmap = LinearSegmentedColormap.from_list(
        "morandi_density",
        morandi_refined,
        N=256
    )

    model_names = ["PINN", "FNN", "CNN", "LSTM"]
    r_key = 0.2

    sns.set_style("white")

    plt.rcParams.update({
        "font.family": "Arial",
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0
    })

    if r_key not in trajectories_dict:
        print(f"Key {r_key} not found.")
        return

    for m_name in model_names:
        all_true, all_pred, all_efc = [], [], []

        for repeat_idx in trajectories_dict[r_key].keys():
            for pair_key in trajectories_dict[r_key][repeat_idx].keys():
                if "cell04_to_cell05" in pair_key:
                    continue

                data = trajectories_dict[r_key][repeat_idx][pair_key]

                if "BASE" in data and m_name in data:
                    all_true.extend(data["BASE"]["actual"])
                    all_pred.extend(data[m_name])
                    all_efc.extend(data["BASE"]["efc"])

        if len(all_true) == 0:
            continue

        all_true = np.asarray(all_true, dtype=float)
        all_pred = np.asarray(all_pred, dtype=float)
        all_efc = np.asarray(all_efc, dtype=float)

        if np.nanmax(all_true) <= 1.5:
            true_pct = all_true * 100
            pred_pct = all_pred * 100
        else:
            true_pct = all_true
            pred_pct = all_pred

        error_pct = pred_pct - true_pct


        fig = plt.figure(figsize=(5.2, 2.35), dpi=300)
        gs = GridSpec(
            1,
            2,
            width_ratios=[1.0, 0.78],
            wspace=0.08
        )

        ax_parity = fig.add_subplot(gs[0])
        ax_density = fig.add_subplot(gs[1])

        axis_min, axis_max = 0, 60
        ticks_pct = [0, 20, 40, 60]

        # =========================
        # Left: parity plot
        # =========================
        ax_parity.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="black",
            linestyle="--",
            lw=1.0,
            zorder=1
        )

        ax_parity.scatter(
            true_pct,
            pred_pct,
            c=all_efc,
            cmap=point_cmap,
            s=7,
            alpha=0.55,
            edgecolor="none",
            zorder=2
        )

        ax_parity.set_xlim(axis_min, axis_max)
        ax_parity.set_ylim(axis_min, axis_max)
        ax_parity.set_xticks(ticks_pct)
        ax_parity.set_yticks(ticks_pct)

        ax_parity.set_xlabel(
            "True capacity loss [%]",
            fontsize=10,
            labelpad=5
        )
        ax_parity.set_ylabel(
            "Predicted capacity loss [%]",
            fontsize=10,
            labelpad=5
        )

        ax_parity.tick_params(
            axis="both",
            labelsize=9,
            width=1.0,
            length=3
        )

        ax_parity.set_title(
            m_name,
            loc="left",
            fontsize=12,
            fontweight="normal",
            pad=4
        )

        for spine in ax_parity.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("black")

        # =========================
        # Right: minimal KDE contour
        # =========================
        try:
      
            if len(true_pct) > 2500:
                np.random.seed(42)
                idx = np.random.choice(len(true_pct), 2500, replace=False)
                plot_true = true_pct[idx]
                plot_error = error_pct[idx]
            else:
                plot_true = true_pct
                plot_error = error_pct

            err_left = 20
            err_right = -20

            # KDE grid
            xmin, xmax = err_right, err_left
            ymin, ymax = axis_min, axis_max

            xx, yy = np.meshgrid(
                np.linspace(xmin, xmax, 180),
                np.linspace(ymin, ymax, 180)
            )

            values = np.vstack([plot_error, plot_true])
            positions = np.vstack([xx.ravel(), yy.ravel()])

    
            kde = gaussian_kde(values, bw_method=0.25)
            zz = kde(positions).reshape(xx.shape)

            zz = (zz - np.nanmin(zz)) / (
                np.nanmax(zz) - np.nanmin(zz) + 1e-12
            )


            ax_density.scatter(
                plot_error,
                plot_true,
                color="#5A6096",
                s=4.5,
                alpha=0.13,
                edgecolor="none",
                zorder=2
            )


            ax_density.axvline(
                0,
                color="black",
                linestyle=":",
                lw=1.1,
                zorder=4
            )

            ax_density.set_xlim(err_left, err_right)
            ax_density.set_xticks([20, 0, -20])

            ax_density.set_ylim(axis_min, axis_max)
            ax_density.set_yticks(ticks_pct)
            ax_density.set_yticklabels([])

            ax_density.set_xlabel(
                "Error [%]",
                fontsize=10,
                labelpad=5
            )

            ax_density.tick_params(
                axis="both",
                labelsize=9,
                width=1.0,
                length=3
            )

            for spine in ax_density.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color("black")

            cbar_ax = ax_density.inset_axes([0.56, 0.82, 0.32, 0.055])

            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=contour_cmap, norm=norm)
            sm.set_array([])

            cb = plt.colorbar(
                sm,
                cax=cbar_ax,
                orientation="horizontal"
            )

            cb.ax.set_title(
                "Density",
                fontsize=8,
                pad=2
            )

            cb.set_ticks([0, 1])
            cb.set_ticklabels(["Low", "High"])

            cb.ax.tick_params(
                labelsize=7,
                length=2,
                width=0.8,
                pad=1
            )

            cb.outline.set_visible(False)

        except Exception as e:
            print(f"Density error for {m_name}: {e}")

        plt.subplots_adjust(
            left=0.12,
            right=0.97,
            bottom=0.24,
            top=0.88
        )

        plt.show()


plot_parity_and_density_minimal(all_trajectories_tgt)
