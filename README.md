# FLEX: Frequency Layer Explanation

## Overview

This repository contains the implementation of **FLEX (Frequency Layer Explanaition)**, a method designed to explain the predictions of deep neural networks (DNNs) for sleep stage classification using EEG data. FLEX combines **Integrated Gradients (IG)** with frequency-domain analysis (via the **Discrete Cosine Transform (DCT)**) to provide frequency-based relevance attributions.

The method is particularly useful for understanding how different frequency components of EEG signals influence the predictions of a DNN, enhancing model interpretability in the context of sleep research.

---

## Features

- **DCT Transformation**: Input EEG signals are transformed into the frequency domain using the DCT.
- **iDCT Integration**: The inverse DCT (iDCT) is implemented as the first layer in the DNN to process frequency-domain inputs.
- **Integrated Gradients Attribution**: Captum's IG method is used to compute relevance scores for frequency bands, providing insights into the features contributing to the model's predictions.

---

## Results

The following plot shows the average weighted frequency attribution for EEG segments classified into different sleep stages: Wake, REM, N1, N2, and N3. Each plot highlights the frequency bands contributing to the classification of each sleep stage.
![Average weighted frequency attribution for EEG segments](./eeg_frequency_attributions_plots/eeg_relevant_plots_1.png)
"Average weighted frequency attribution for EEG segments classified into different sleep stages: Wake, REM, N1, N2, and N3. Each plot highlights the frequency bands contributing to the classification of each sleep stage."

---

## Installation

### Requirements
- Python 3.8+
- Required libraries:
 - `scipy`
 - `torch`
 - `captum`
 - `torch-dct`

### Install Dependencies
You can install the required Python libraries using `pip`:
```bash
pip install scipy torch captum torch-dct
