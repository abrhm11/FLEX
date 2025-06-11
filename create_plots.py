import os
import glob
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Configuration ---
data_dir     = 'all_ig_data'
output_dir   = 'aggregated_plots'
indiv_dir    = 'individual_patient_plots'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(indiv_dir, exist_ok=True)

labels_list     = [0, 1, 2, 3, 4]
labels_plot_title = ["Wake", "REM", "N1", "N2", "N3"]
sampling_rate   = 100  # Hz

def compute_mean_by_label(labels, attributions, labels_list):
    means = {}
    for lbl in labels_list:
        mask = (labels == lbl)
        if np.any(mask):
            means[lbl] = attributions[mask].mean(axis=0)
        else:
            means[lbl] = np.zeros(attributions.shape[1], dtype=attributions.dtype)
    return means

# --- Load data ---
filepaths = sorted(glob.glob(os.path.join(data_dir, 'patient_fft_ig_*.npz')))
patient_data = []
individual_means = []

for fp in filepaths:
    data = np.load(fp)
    labels = data['labels']
    attributions = data['attributions']
    patient_id = os.path.splitext(os.path.basename(fp))[0].split('_')[-1]
    patient_data.append((patient_id, labels, attributions))
    individual_means.append(compute_mean_by_label(labels, attributions, labels_list))
    data.close()

# --- Frequency axis ---
segment_length = patient_data[0][2].shape[1]
freqs = np.arange(segment_length) * sampling_rate / (segment_length)

# --- Aggregate from individual means ---
mean_attrs = {}
for lbl in labels_list:
    mean_stack = [ind_means[lbl] for ind_means in individual_means]
    mean_attrs[lbl] = np.vstack(mean_stack).mean(axis=0)

# --- For aggregated attributions ---

# --- Normalize color scale ---
all_values = np.concatenate([v for v in mean_attrs.values()])
min_attr, max_attr = all_values.min(), all_values.max()
norm = mpl.colors.Normalize(vmin=min_attr, vmax=max_attr)

# --- Extra space in y-Axis ---
ylim_aggr = 0.1 * max(abs(min_attr), abs(max_attr))

# --- Plot aggregate ---
agg_pdf_path = os.path.join(output_dir, 'all_patients_fft_mean_attributions.pdf')
with PdfPages(agg_pdf_path) as pdf:
    fig, axs = plt.subplots(len(labels_list), 1, figsize=(6, 10), sharex=True)
    for ax, lbl in zip(axs, labels_list):
        norm_lbl = mpl.colors.Normalize(vmin=mean_attrs[lbl].min(), vmax=mean_attrs[lbl].max())
        ax.bar(freqs, mean_attrs[lbl], color=plt.cm.coolwarm(norm_lbl(mean_attrs[lbl])), width=0.3)
        for vline in [2, 8, 12.5]:
            ax.axvline(x=vline, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(labels_plot_title[lbl] + " Sleep Stage")
        ax.set_xlim(0, 50)
        ax.set_ylim(min_attr - ylim_aggr, max_attr + ylim_aggr)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.xaxis.set_ticks(np.linspace(0, 50, 6))
        ax.tick_params(axis="both", labelsize=10)
        if lbl == labels_list[-1]:
            ax.set_xlabel("Frequency (Hz)", fontsize=12)
    axs[2].set_ylabel('Mean Attribution', fontsize=12)
    
    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', shrink=0.8, pad=1)
    cbar.ax.tick_params(axis="both", right=False, labelright=False)

    # Colorbar Labels
    x_offset = cbar.ax.get_position().x1 - 0.04
    fig.text(x_offset, cbar.ax.get_position().y1, "Pro", ha="left", va="center", fontsize=10, transform=fig.transFigure, color=(0.7057, 0.0156, 0.1502, 1.0))
    fig.text(x_offset, (cbar.ax.get_position().y0 + cbar.ax.get_position().y1) / 2, "Irr", ha="left", va="center", fontsize=10, transform=fig.transFigure, color=(0.5, 0.5, 0.5, 1.0))
    fig.text(x_offset, cbar.ax.get_position().y0, "Con", ha="left", va="center", fontsize=10, transform=fig.transFigure, color=(0.2298, 0.2987, 0.7537, 1.0))

    fig.subplots_adjust(right=0.76, hspace=0.4)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
print(f'Aggregated plot saved to {agg_pdf_path}')

# --- Plot individuals ---
sample_data = random.sample(list(zip(patient_data, individual_means)), min(10, len(patient_data)))
for ((patient_id, labels, attributions), mean_lbl) in sample_data:
    out_path = os.path.join(indiv_dir, f'patient_fft_{patient_id}_mean_attributions.pdf')
    # --- Normalize color scale ---
    all_individual_values = np.concatenate([v for v in mean_lbl.values()])
    individual_min_attr, individual_max_attr = all_individual_values.min(), all_individual_values.max()
    individual_norm = mpl.colors.Normalize(vmin=individual_min_attr, vmax=individual_max_attr)
    
    # --- Extra space in y-Axis ---
    ylim_individual = 0.1 * max(abs(individual_min_attr), abs(individual_max_attr))
    with PdfPages(out_path) as pdf:
        fig, axs = plt.subplots(len(labels_list), 1, figsize=(6, 10), sharex=True)
        for ax, lbl in zip(axs, labels_list):
            norm_lbl=mpl.colors.Normalize(vmin=mean_lbl[lbl].min(), vmax=mean_lbl[lbl].max())
            ax.bar(freqs, mean_lbl[lbl], color=plt.cm.coolwarm(norm_lbl(mean_lbl[lbl])), width=0.3)
            for vline in [2, 8, 12.5]:
                ax.axvline(x=vline, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_title(labels_plot_title[lbl] + " Sleep Stage")
            ax.set_xlim(0, 50)
            ax.set_ylim(individual_min_attr - ylim_individual, individual_max_attr + ylim_individual)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.xaxis.set_ticks(np.linspace(0, 50, 6))
            ax.tick_params(axis="both", labelsize=10)
            if lbl == labels_list[-1]:
                ax.set_xlabel("Frequency (Hz)", fontsize=12)
        axs[2].set_ylabel('Mean Attribution', fontsize=12)

        sm = mpl.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=individual_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical', shrink=0.8, pad=1)
        cbar.ax.tick_params(axis="both", right=False, labelright=False)

        x_offset = cbar.ax.get_position().x1 - 0.04
        fig.text(x_offset, cbar.ax.get_position().y1, "Pro", ha="left", va="center", fontsize=10, transform=fig.transFigure, color=(0.7057, 0.0156, 0.1502, 1.0))
        fig.text(x_offset, (cbar.ax.get_position().y0 + cbar.ax.get_position().y1) / 2, "Irr", ha="left", va="center", fontsize=10, transform=fig.transFigure, color=(0.5, 0.5, 0.5, 1.0))
        fig.text(x_offset, cbar.ax.get_position().y0, "Con", ha="left", va="center", fontsize=10, transform=fig.transFigure, color=(0.2298, 0.2987, 0.7537, 1.0))

        fig.subplots_adjust(right=0.76, hspace=0.4)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    print(f'Saved individual plot for patient {patient_id} → {out_path}')