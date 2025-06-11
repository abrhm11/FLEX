import os
import glob
import random
import numpy as np
import torch
from collections import OrderedDict
from captum.attr import IntegratedGradients

# --- Load and prepare model ---
from XAI_EEG_data.DRCNN import Sleep_model_MultiTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Sleep_model_MultiTarget(
    numSignals=1,
    binClasses=[5],
    dilationLayers=[1,2,4,8,16,8,4,2,1],
    channelMultiplier=2,
    kernelSize=20,
    useSkipLLSTM=True,
    lstmChannels=128,
    unitMultiplierMod=4,
    unitMultiplierDS=3,
    downSampleSteps=[2,5,5,5,12],
    skipMarginalize=True,
    useNormalizer=False,
    batchSize=2,
)
# Load model weights
state = np.load('XAI_EEG_data/model_state.npz', allow_pickle=True)['model_state'].item()
sd = OrderedDict((k, v) for k, v in state.items())
model.load_state_dict(sd)
model.eval().to(device)

# Initialize Integrated Gradients
ig = IntegratedGradients(model)

# Create output directory
out_dir = 'all_ig_data'
os.makedirs(out_dir, exist_ok=True)

# Parameters
segment_length = 3000
n_steps_ig     = 3
# Initialize patient ID
patient_id = 1

# Iterate over all patient files
for filepath in glob.glob('eeg_patient_data/patient_data_*.npz')[:2]:
    # Load data
    data = np.load(filepath, allow_pickle=True)
    x_full = data['x']  # raw EEG signal
    y_onehot = data['y']
    # Convert labels to integer classes
    labels = np.array([int(np.argmax(lbl)) for lbl in y_onehot], dtype=np.int8)

    # Split signal into segments
    num_segments = len(x_full) // segment_length
    segments = np.array_split(x_full[:num_segments * segment_length], num_segments)

    # Container for attributions
    attributions = np.zeros((num_segments, segment_length // 2 + 1), dtype=np.float32)

    random.seed(42)
    # Compute IG for each segment
    for idx, seg in enumerate(random.sample(segments, 10)):
        seg_t = torch.from_numpy(seg.squeeze()).to(device)

        # FFT
        seg_fft = torch.fft.fft(seg_t.squeeze())
        N = seg_fft.shape[0]
        seg_fft_pos = seg_fft[: N // 2 + 1]

        inp      = seg_fft_pos
        baseline = torch.zeros_like(inp).to(device)

        # Compute attributions
        attr = ig.attribute(
            inputs=inp,
            baselines=baseline,
            target=(labels[idx], idx),
            additional_forward_args=(
                torch.unsqueeze(torch.from_numpy(x_full).to(device), 0).permute(0,2,1),
                idx,
                segment_length
            ),
            n_steps=n_steps_ig
        )
        # Normalize and store
        attr_np = attr.detach().cpu().numpy().flatten().real
        attributions[idx] = attr_np

    # Save result per patient
    out_path = os.path.join(out_dir, f'patient_fft_ig_{patient_id}.npz')
    np.savez(
        out_path,
        labels=labels[:num_segments],
        attributions=attributions
    )
    print(f"Saved IG data for patient {patient_id} → {out_path}")

    # Free memory
    del data, x_full, y_onehot, segments, attributions, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    patient_id += 1