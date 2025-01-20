#Import necessary libraries and the model
import numpy as np
import torch
import torch_dct
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import dct
from captum.attr import IntegratedGradients
from scipy.io import savemat
from matplotlib.colors import LinearSegmentedColormap

#os.chdir('/content/gdrive/MyDrive/Jupyter_Notebooks_KDD')
from XAI_EEG_data.DRCNN import Sleep_model_MultiTarget

#Checking if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

#Load the model using this fine-tuned parameters
model = Sleep_model_MultiTarget(
    numSignals=1,
    binClasses=[5],
    dilationLayers=[1, 2, 4, 8, 16, 8, 4, 2, 1],
    channelMultiplier=2,
    kernelSize=20,
    useSkipLLSTM=True,##
    lstmChannels=128,
    unitMultiplierMod=4,
    unitMultiplierDS=3,
    downSampleSteps=[2, 5, 5, 5, 12] ,
    skipMarginalize=True,
    useNormalizer=False,##
    batchSize=2,
)

path = 'XAI_EEG_data'
#Loading the weights
model_state = np.load(os.path.join(path, 'model_state.npz'), allow_pickle=True)
model_state_dict = model_state['model_state'].item()

state_dict = OrderedDict()
for k, v in model_state_dict.items():
    state_dict[k] = v

#Load the state dictionary into your model
model.load_state_dict(state_dict)

#Set the model to evaluation mode
model.eval()
model.to(device);

#Load as a first step the EEG data from a patient
path_all_data = 'eeg_patient_data'

indices_0 = []
indices_1 = []
indices_2 = []
indices_3 = []
indices_4 = []

all_total_data = []

all_segments = []

for i in range(1,10):
    # Indices der Klasses vorbereiten
    patient_data = np.load(os.path.join(path_all_data, f'patient_data_{i}.npz'))
    y = patient_data['y']
    y = [np.argmax(j) for j in y]

    # Indices der Klassen in den Segmenten
    indices_0.append([index for index, value in enumerate(y) if value == 0]) # enthält arrays der einzelnen patienten, welche die zugehörigen indices der Segmente der Schlafphase 0 (wach) enthalten
    indices_1.append([index for index, value in enumerate(y) if value == 1])
    indices_2.append([index for index, value in enumerate(y) if value == 2])
    indices_3.append([index for index, value in enumerate(y) if value == 3])
    indices_4.append([index for index, value in enumerate(y) if value == 4])

    # komplette Schlafdaten vorbereiten
    x = patient_data['x']
    # Segmente vorbereiten
    segment_length = 3000
    # Anzahl der Segmente berechnen
    num_segments = len(x) // segment_length
    # Array in Segmente aufteilen
    x_segments = np.array(np.split(x[:num_segments * segment_length], num_segments))
    x = torch.from_numpy(x).to(device)
    x = torch.unsqueeze(x, 0)
    x = x.permute(0, 2, 1)
    all_total_data.append(x) # enthält arrays der einzelnen patienten, welche die zugehörigen kompleten EEG-Daten/-Aufzeichnungen darstellen 

    # all_segments[patient][segments]
    all_segments.append(x_segments) # enthält arrays der einzelnen patienten, welche die zugehörigen arrays der Segment

# Liste aller indices-Arrays, die verarbeitet werden sollen
all_indices = [indices_0, indices_1, indices_2, indices_3, indices_4]

# Define the folder path
folder_path = "eeg_frequencies_relevance_plots"
os.makedirs(folder_path, exist_ok=True)

pdf_file = PdfPages(os.path.join(folder_path, "eeg_frequencies_relevance_plots_1.pdf"))

model.train()
ig = IntegratedGradients(model) # get an instance using our model

internal_batch_size = 10

# plots and values for single segments - later segments + surrounding segments

# Liste, um die durchschnittlichen (positiv gewichteten) Frequenzen für jedes indices-Array zu speichern
overall_average_frequencies_pos_list = []
overall_average_frequencies_neg_list = []
mean_attributions_list = []
all_frequencies = None

phases = ["wake", "REM", "N1", "N2", "N3"]

cmap = plt.cm.RdYlGn

def non_linear_colormap(name, colors, power=0.1):
    # Erzeugt eine Colormap mit einem nicht-linearen Übergang.
    # Invertierung der Farbfolge und Kontrolle der Transformation durch 'power'.
    index = np.linspace(0, 1, 256) ** power  # Nicht-lineares Index-Mapping
    cdict = {
        'red':   [(i, colors[0][0], colors[0][0]) if i == 0 else (i, colors[1][0], colors[1][0])
                  for i in index],
        'green': [(i, colors[0][1], colors[0][1]) if i == 0 else (i, colors[1][1], colors[1][1])
                  for i in index],
        'blue':  [(i, colors[0][2], colors[0][2]) if i == 0 else (i, colors[1][2], colors[1][2])
                  for i in index]
    }
    return LinearSegmentedColormap(name, segmentdata=cdict, N=256)

# Definition der Colormaps: LightGray zu Green und zu Red
cmap_pos = non_linear_colormap('GreensToGray', [(0.827, 0.827, 0.827), (0.0, 0.36, 0.0)])
cmap_neg = non_linear_colormap('RedsToGray', [(0.827, 0.827, 0.827), (0.36, 0.0, 0.0)])

# Zielordner für mean_attributions Matrix
output_folder = "mean_attributions_single_segment"
os.makedirs(output_folder, exist_ok=True)

# Äußere Schleife über jedes indices-Array, das eine Schlafphase repräsentiert
for i, indices_i in enumerate(all_indices):
    print(f"Processing segments of {phases[i]} stage...")  # Ausgabe für das aktuelle indices-Array

    #all_weighted_pos_attributions = []
    #all_weighted_neg_attributions = []
    all_weighted_attributions = []
    all_amplitudes = []
    
    # Innere Schleife über patients
    for k in range(len(indices_i)):
        indices_i_k = indices_i[k] # indices of i'th sleep stage and k'th patient
        x_segments = all_segments[k]
        segments_i = x_segments[indices_i_k[:len(indices_i_k)]]
        x = all_total_data[k]
    
        # Innere Schleife über jeden Index im aktuellen indices-Array (Segmente)
        for j in range(len(indices_i_k)):
            sleep_phase = y[indices_i_k[j]]  # Definiere den sleep_phase für den aktuellen Index
            target = (sleep_phase, indices_i_k[j])
            x_segment = segments_i[j]
            
            # Führe die DCT auf das Segment durch
            x_segment_dct = dct(x_segment.squeeze(), norm='ortho')
            
            # Schneide die irrelevanten Frequenzen ab
            cut_segment = int(segment_length * 0.3)
            x_segment_dct[cut_segment:] = 0
            
            # Konvertiere das Ergebnis zurück in einen PyTorch-Tensor
            x_segment_dct = torch.tensor(x_segment_dct).to(device)

            baseline = torch.zeros_like(x_segment_dct).to(device) # Baseline tensor representing abense of features, we use zeros
            
            # Berechne die Attributionswerte für den aktuellen Index
            ig_relevances = ig.attribute(inputs=x_segment_dct, 
                                         baselines=baseline, 
                                         target=target, 
                                         additional_forward_args=(x, indices_i_k[j], segment_length), 
                                         internal_batch_size=internal_batch_size, 
                                         n_steps=50)
            
            # Konvertiere zu NumPy-Array und schneide auf die relevanten Frequenzen zu
            attributions_magnitude = ig_relevances.cpu().detach().numpy().flatten()
            
            # Frequenzen für die DCT berechnen
            frequencies = np.fft.fftfreq(len(x_segment_dct), d=1/100)
            
            # Positive und negative Attributionswerte normalisieren
            #attributions_magnitude_pos = np.where(attributions_magnitude > 0, attributions_magnitude, 0)
            #attributions_magnitude_neg = np.where(attributions_magnitude < 0, attributions_magnitude, 0)
            
            #sum_attributions_pos = np.sum(attributions_magnitude_pos)
            #sum_attributions_neg = np.sum(attributions_magnitude_neg)

            sum_attributions = np.sum(np.abs(attributions_magnitude))
            
            # Relativen Attributionswert pro Frequenz berechnen und speichern
            if all_frequencies is None:
                all_frequencies = frequencies
            
            #relative_pos_attributions = attributions_magnitude_pos / sum_attributions_pos
            #relative_neg_attributions = attributions_magnitude_neg / sum_attributions_neg
            
            #all_weighted_pos_attributions.append(relative_pos_attributions)
            #all_weighted_neg_attributions.append(relative_neg_attributions)

            relative_attributions = attributions_magnitude/sum_attributions

            all_weighted_attributions.append(relative_attributions)
    
            # Amplituden speichern
            amplitudes = np.abs(x_segment_dct).cpu().detach().numpy()
            all_amplitudes.append(amplitudes)
    
    # Mittelwerte der Attributionswerte über alle Segmente hinweg für jede Frequenz berechnen (sollte eigentlich auch gewichtet sein, unterscheidet sich aber hier nur in einem Faktor)
    #mean_weighted_pos_attributions = np.mean(all_weighted_pos_attributions, axis=0)
    #mean_weighted_neg_attributions = np.mean(all_weighted_neg_attributions, axis=0)
    #mean_weighted_attributions = mean_weighted_pos_attributions + mean_weighted_neg_attributions
    mean_attributions = np.mean(all_weighted_attributions, axis=0)

    mean_weighted_pos_attributions = np.where(mean_attributions > 0, mean_attributions, 0)
    mean_weighted_neg_attributions = np.where(mean_attributions < 0, mean_attributions, 0)

    # Mittelwerte der Amplituden über alle Segmente hinweg für jede Frequenz berechnen
    mean_amplitudes = np.mean(all_amplitudes, axis=0)
    
    # Gewichteter Durchschnitt der Frequenzen berechnen
    overall_average_frequency_pos = np.sum(all_frequencies * mean_weighted_pos_attributions) / np.sum(mean_weighted_pos_attributions)
    overall_average_frequency_neg = np.sum(all_frequencies * mean_weighted_neg_attributions) / np.sum(mean_weighted_neg_attributions)

    # Speichern der durchschnittlichen Frequenzen
    overall_average_frequencies_pos_list.append(overall_average_frequency_pos)
    overall_average_frequencies_neg_list.append(overall_average_frequency_neg)
    mean_attributions_list.append(mean_attributions)
    
    # Plot 1: Average Amplitudes colored by weighted attributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Positive and negative color map for attribution values
    cmap = plt.cm.RdYlGn
    
    # First Plot - Average Amplitudes
    bars = ax1.bar(
        all_frequencies,
        mean_amplitudes,
        color=cmap(((mean_attributions / np.max(np.abs(mean_attributions)))+1)/2),
        label="coloured by weighted attributions",
        width=0.2
    )
    
    # Settings for the first plot
    ax1.set_xlabel("Frequency (Hz)", fontsize=20)
    ax1.set_ylabel("Average Amplitudes", fontsize=20)
    ax1.set_title(f"Average weighted attributions of frequencies in {phases[i]} stage", fontsize=22)
    ax2.tick_params(axis='both', labelsize=18)
    ax1.set_xlim(0, 40)  # Limit to 40 Hz
    ax1.set_ylim(0, np.max(mean_amplitudes) * 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Average Weighted Positive and Negative Attributions
    
    # Positive weighted attributions
    bars_pos = ax2.bar(
        all_frequencies,
        mean_weighted_pos_attributions,
        color=cmap_pos(mean_weighted_pos_attributions / np.max(np.abs(mean_attributions))),
        label="Positively weighted attributions",
        width=0.2
    )
        
    # Negative weighted attributions (negated for visual clarity)
    bars_neg = ax2.bar(
        all_frequencies,
        mean_weighted_neg_attributions,
        color=cmap_neg(-mean_weighted_neg_attributions / np.max(np.abs(mean_attributions))),
        label="Negatively weighted attributions",
        width=0.2
    )
    
    # Settings for the second plot
    ax2.set_xlabel("Frequency (Hz)", fontsize=20)
    ax2.set_ylabel("Average Attribution", fontsize=20)
    ax2.set_title(f"Average weighted positive/negative attributions in {phases[i]} stage", fontsize=22)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_xlim(0, 40)
    ax2.set_ylim(np.min(mean_weighted_neg_attributions) * 1.1, np.max(mean_weighted_pos_attributions) * 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the figure with both plots stacked vertically
    plt.tight_layout()
    pdf_file.savefig()
    plt.show()

    # Dateiname erstellen
    mat_filename = f"mean_attributions_single_segment_{phases[i]}.mat"
    mat_filepath = os.path.join(output_folder, mat_filename)
    
    # Speichere die Liste als Matrix
    savemat(mat_filepath, {"mean_attributions": mean_attributions})

# Now let's add the printed output to the PDF
fig_text = plt.figure(figsize=(8, 6))  # Create a new figure for text
plt.axis('off')  # Turn off the axis

# Prepare the text output
text_output = "\nAverage (positiv weighted) frequencies for all sleep stages:\n"
for i, avg_freq in enumerate(overall_average_frequencies_pos_list):
    text_output += f"{phases[i]}: {avg_freq:.2f} Hz\n"

text_output += "\nAverage (negativ weighted) frequencies for all sleep stages:\n"
for i, avg_freq in enumerate(overall_average_frequencies_neg_list):
    text_output += f"{phases[i]}: {avg_freq:.2f} Hz\n"

text_output += "\nMost dominant (positive or negative attributed) frequencies compared to the other sleep stages:\n"
sum_mean_attributions_list = np.sum(np.abs(mean_attributions_list))
for i, attr in enumerate(mean_attributions_list):
    attr_arg = np.argmax(np.abs(attr)/sum_mean_attributions_list)
    frequency = all_frequencies[attr_arg]
    text_output += f"{phases[i]}: {frequency:.2f} Hz at (relative) attribution {attr[attr_arg]:.2f}\n"

# Add the text to the figure
plt.text(0.5, 0.5, text_output, fontsize=12, ha='center', va='center')

# Save the text figure to the PDF
pdf_file.savefig(fig_text)

# Close the PDF file
pdf_file.close()






# plots and values for segments + surrounding segments

folder_path = "eeg_frequencies_relevance_plots_surrounding_segments"
os.makedirs(folder_path, exist_ok=True)

pdf_file = PdfPages(os.path.join(folder_path, "eeg_frequencies_relevance_plots_surrounding_segments_1.pdf"))

# Liste, um die durchschnittlichen (positiv gewichteten) Frequenzen für jedes indices-Array zu speichern
overall_average_frequencies_pos_list = []
overall_average_frequencies_neg_list = []
mean_attributions_list = []
all_frequencies = None

# Zielordner für mean_attributions Matrix
output_folder = "mean_attributions_segment_and_surrounding"
os.makedirs(output_folder, exist_ok=True)

# Äußere Schleife über jedes indices-Array, das eine Schlafphase repräsentiert
for i, indices_i in enumerate(all_indices):
    print(f"Processing segments of {phases[i]} stage...")  # Ausgabe für das aktuelle indices-Array

    #all_weighted_pos_attributions = []
    #all_weighted_neg_attributions = []
    all_weighted_attributions = []
    all_amplitudes = []
    
    # Innere Schleife über patients
    for k in range(len(indices_i)):
        indices_i_k = indices_i[k] # indices of i'th sleep stage and k'th patient
        x_segments = all_segments[k]
        segments_i = x_segments[indices_i_k[:len(indices_i_k)]]
        x = all_total_data[k]
    
        # Innere Schleife über jeden Index im aktuellen indices-Array (Segmente)
        for j in range(1,len(indices_i_k-1)):
            if indices_i_k[j-1] == indices_i_k[j]-1 and indices_i_k[j+1] == indices_i_k[j]+1:
                sleep_phase = y[indices_i_k[j]]  # Definiere den sleep_phase für den aktuellen Index
                target = (sleep_phase, indices_i_k[j])
                x_segment = np.concatenate((segments_i[j-1], segments_i[j], segments_i[j+1]), axis = 0)
                segment_length = len(x_segment)
                
                # Führe die DCT auf das Segment durch
                x_segment_dct = dct(x_segment.squeeze(), norm='ortho')
                
                # Schneide die irrelevanten Frequenzen ab
                cut_segment = int(segment_length * 0.3)
                x_segment_dct[cut_segment:] = 0
                
                # Konvertiere das Ergebnis zurück in einen PyTorch-Tensor
                x_segment_dct = torch.tensor(x_segment_dct).to(device)
    
                baseline = torch.zeros_like(x_segment_dct).to(device) # Baseline tensor representing abense of features, we use zeros
                
                # Berechne die Attributionswerte für den aktuellen Index
                ig_relevances = ig.attribute(inputs=x_segment_dct, 
                                             baselines=baseline, 
                                             target=target, 
                                             additional_forward_args=(x, indices_i_k[j-1], segment_length), 
                                             internal_batch_size=internal_batch_size, 
                                             n_steps=50)
                
                # Konvertiere zu NumPy-Array und schneide auf die relevanten Frequenzen zu
                attributions_magnitude = ig_relevances.cpu().detach().numpy().flatten()
                
                # Frequenzen für die DCT berechnen
                frequencies = np.fft.fftfreq(len(x_segment_dct), d=1/100)
                
                # Positive und negative Attributionswerte normalisieren
                #attributions_magnitude_pos = np.where(attributions_magnitude > 0, attributions_magnitude, 0)
                #attributions_magnitude_neg = np.where(attributions_magnitude < 0, attributions_magnitude, 0)
                
                #sum_attributions_pos = np.sum(attributions_magnitude_pos)
                #sum_attributions_neg = np.sum(attributions_magnitude_neg)
    
                sum_attributions = np.sum(np.abs(attributions_magnitude))
                
                # Relativen Attributionswert pro Frequenz berechnen und speichern
                if all_frequencies is None:
                    all_frequencies = frequencies
                
                #relative_pos_attributions = attributions_magnitude_pos / sum_attributions_pos
                #relative_neg_attributions = attributions_magnitude_neg / sum_attributions_neg
                
                #all_weighted_pos_attributions.append(relative_pos_attributions)
                #all_weighted_neg_attributions.append(relative_neg_attributions)
    
                relative_attributions = attributions_magnitude/sum_attributions
    
                all_weighted_attributions.append(relative_attributions)
        
                # Amplituden speichern
                amplitudes = np.abs(x_segment_dct).cpu().detach().numpy()
                all_amplitudes.append(amplitudes)

    if not all_weighted_attributions:
        print('Empty list')
    else:
        # Mittelwerte der Attributionswerte über alle Segmente hinweg für jede Frequenz berechnen (sollte eigentlich auch gewichtet sein, unterscheidet sich aber hier nur in einem Faktor)
        #mean_weighted_pos_attributions = np.mean(all_weighted_pos_attributions, axis=0)
        #mean_weighted_neg_attributions = np.mean(all_weighted_neg_attributions, axis=0)
        #mean_weighted_attributions = mean_weighted_pos_attributions + mean_weighted_neg_attributions
        mean_attributions = np.mean(all_weighted_attributions, axis=0)
    
        mean_weighted_pos_attributions = np.where(mean_attributions > 0, mean_attributions, 0)
        mean_weighted_neg_attributions = np.where(mean_attributions < 0, mean_attributions, 0)
    
        # Mittelwerte der Amplituden über alle Segmente hinweg für jede Frequenz berechnen
        mean_amplitudes = np.mean(all_amplitudes, axis=0)
        
        # Gewichteter Durchschnitt der Frequenzen berechnen
        overall_average_frequency_pos = np.sum(all_frequencies * mean_weighted_pos_attributions) / np.sum(mean_weighted_pos_attributions)
        overall_average_frequency_neg = np.sum(all_frequencies * mean_weighted_neg_attributions) / np.sum(mean_weighted_neg_attributions)
    
        # Speichern der durchschnittlichen Frequenzen
        overall_average_frequencies_pos_list.append(overall_average_frequency_pos)
        overall_average_frequencies_neg_list.append(overall_average_frequency_neg)
        mean_attributions_list.append(mean_attributions)
        
        # Plot 1: Average Amplitudes colored by weighted attributions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        
        # First Plot - Average Amplitudes
        bars = ax1.bar(
            all_frequencies,
            mean_amplitudes,
            color=cmap(((mean_attributions / np.max(np.abs(mean_attributions)))+1)/2),
            label="coloured by weighted attributions",
            width=0.2
        )
        
        # Settings for the first plot
        ax1.set_xlabel("Frequency (Hz)", fontsize=20)
        ax1.set_ylabel("Average Amplitudes", fontsize=20)
        ax1.set_title(f"Average weighted attributions of frequencies in {phases[i]} stage", fontsize=22)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.set_xlim(0, 40)  # Limit to 40 Hz
        ax1.set_ylim(0, np.max(mean_amplitudes) * 1.1)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
        # Plot 2: Average Weighted Positive and Negative Attributions    
        
        # Positive weighted attributions
        bars_pos = ax2.bar(
            all_frequencies,
            mean_weighted_pos_attributions,
            color=cmap_pos(mean_weighted_pos_attributions / np.max(np.abs(mean_attributions))),
            label="Positively weighted attributions",
            width=0.2
        )
            
        # Negative weighted attributions (negated for visual clarity)
        bars_neg = ax2.bar(
            all_frequencies,
            mean_weighted_neg_attributions,
            color=cmap_neg(-mean_weighted_neg_attributions / np.max(np.abs(mean_attributions))),
            label="Negatively weighted attributions",
            width=0.2
        )
        
        # Settings for the second plot
        ax2.set_xlabel("Frequency (Hz)", fontsize=20)
        ax2.set_ylabel("Average Attribution", fontsize=20)
        ax2.set_title(f"Average weighted positive/negative attributions in {phases[i]} stage", fontsize=22)
        ax2.tick_params(axis='both', labelsize=18)
        ax2.set_xlim(0, 40)
        ax2.set_ylim(np.min(mean_weighted_neg_attributions) * 1.1, np.max(mean_weighted_pos_attributions) * 1.1)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
        # Show the figure with both plots stacked vertically
        plt.tight_layout()
        pdf_file.savefig()
        plt.show()

        # Dateiname erstellen
        mat_filename = f"mean_attributions_segment_and_surrounding_{phases[i]}.mat"
        mat_filepath = os.path.join(output_folder, mat_filename)
        
        # Speichere die Liste als Matrix
        savemat(mat_filepath, {"mean_attributions": mean_attributions})

if len(overall_average_frequencies_pos_list) == 5 and len(overall_average_frequencies_neg_list) == 5:   
    # Now let's add the printed output to the PDF
    fig_text = plt.figure(figsize=(8, 6))  # Create a new figure for text
    plt.axis('off')  # Turn off the axis
    
    # Prepare the text output
    text_output = "\nAverage (positiv weighted) frequencies for all sleep stages:\n"
    for i, avg_freq in enumerate(overall_average_frequencies_pos_list):
        text_output += f"{phases[i]}: {avg_freq:.2f} Hz\n"
    
    text_output += "\nAverage (negativ weighted) frequencies for all sleep stages:\n"
    for i, avg_freq in enumerate(overall_average_frequencies_neg_list):
        text_output += f"{phases[i]}: {avg_freq:.2f} Hz\n"
    
    text_output += "\nMost dominant (positive or negative attributed) frequencies compared to the other sleep stages:\n"
    sum_mean_attributions_list = np.sum(np.abs(mean_attributions_list))
    for i, attr in enumerate(mean_attributions_list):
        attr_arg = np.argmax(attr/sum_mean_attributions_list)
        frequency = all_frequencies[np.argmax(attr/sum_mean_attributions_list)]
        text_output += f"{phases[i]}: {frequency:.2f} Hz at (relative) attribution {attr[attr_arg]:.2f}\n"
    
    # Add the text to the figure
    plt.text(0.5, 0.5, text_output, fontsize=12, ha='center', va='center')
    
    # Save the text figure to the PDF
    pdf_file.savefig(fig_text)

# Close the PDF file
pdf_file.close()






# plots and values for mean of segments and their direct neighbor segments

# Define the folder path
folder_path = "eeg_frequencies_relevance_plots_mean_of_surrounding_segments"
os.makedirs(folder_path, exist_ok=True)

pdf_file = PdfPages(os.path.join(folder_path, "eeg_frequencies_relevance_plots_mean_of_surrounding_segments_1.pdf"))

# Liste, um die durchschnittlichen (positiv gewichteten) Frequenzen für jedes indices-Array zu speichern
overall_average_frequencies_pos_list = []
overall_average_frequencies_neg_list = []
mean_attributions_list = []
all_frequencies = None

phases = ["wake", "REM", "N1", "N2", "N3"]

# Zielordner für mean_attributions Matrix
output_folder = "mean_attributions_mean_of_surrounding_segments"
os.makedirs(output_folder, exist_ok=True)

# Äußere Schleife über jedes indices-Array, das eine Schlafphase repräsentiert
for i, indices_i in enumerate(all_indices):
    print(f"Processing segments of {phases[i]} stage...")  # Ausgabe für das aktuelle indices-Array

    #all_weighted_pos_attributions = []
    #all_weighted_neg_attributions = []
    all_weighted_attributions = []
    all_amplitudes = []
    
    # Innere Schleife über patients
    for k in range(len(indices_i)):
        indices_i_k = indices_i[k] # indices of i'th sleep stage and k'th patient
        x_segments = all_segments[k]
        segments_i = x_segments[indices_i_k[:len(indices_i_k)]]
        x = all_total_data[k]
    
        # Innere Schleife über jeden Index im aktuellen indices-Array (Segmente)
        for j in range(1, len(indices_i_k)):
            if indices_i_k[j-1] == indices_i_k[j]-1 and indices_i_k[j+1] == indices_i_k[j]+1:
                list_attributions_magnitude = []
                for l in range(3):
                    sleep_phase = y[indices_i_k[j]]  # Definiere den sleep_phase für den aktuellen Index
                    target = (sleep_phase, indices_i_k[j])
                    x_segment = segments_i[j+l-1]
                    segment_length = len(x_segment)
                    
                    # Führe die DCT auf das Segment durch
                    x_segment_dct = dct(x_segment.squeeze(), norm='ortho')
                    
                    # Schneide die irrelevanten Frequenzen ab
                    cut_segment = int(segment_length * 0.3)
                    x_segment_dct[cut_segment:] = 0
                    
                    # Konvertiere das Ergebnis zurück in einen PyTorch-Tensor
                    x_segment_dct = torch.tensor(x_segment_dct).to(device)
        
                    baseline = torch.zeros_like(x_segment_dct).to(device) # Baseline tensor representing abense of features, we use zeros
                    
                    # Berechne die Attributionswerte für den aktuellen Index
                    ig_relevances = ig.attribute(inputs=x_segment_dct, 
                                                 baselines=baseline, 
                                                 target=target, 
                                                 additional_forward_args=(x, indices_i_k[j+l-1], segment_length), 
                                                 internal_batch_size=internal_batch_size, 
                                                 n_steps=50)
                    
                    # Konvertiere zu NumPy-Array und schneide auf die relevanten Frequenzen zu
                    attributions_magnitude = ig_relevances.cpu().detach().numpy().flatten()
                    list_attributions_magnitude.append(attributions_magnitude)

                # Durchschnitt der Frequenzen des Segments und der direkten Nachbar-Segmenten
                attributions_magnitude = np.mean(list_attributions_magnitude, axis=0)
                
                # Frequenzen für die DCT berechnen
                frequencies = np.fft.fftfreq(len(x_segment_dct), d=1/100)
                
                # Positive und negative Attributionswerte normalisieren
                #attributions_magnitude_pos = np.where(attributions_magnitude > 0, attributions_magnitude, 0)
                #attributions_magnitude_neg = np.where(attributions_magnitude < 0, attributions_magnitude, 0)
                
                #sum_attributions_pos = np.sum(attributions_magnitude_pos)
                #sum_attributions_neg = np.sum(attributions_magnitude_neg)
    
                sum_attributions = np.sum(np.abs(attributions_magnitude))
                
                # Relativen Attributionswert pro Frequenz berechnen und speichern
                if all_frequencies is None:
                    all_frequencies = frequencies
                
                #relative_pos_attributions = attributions_magnitude_pos / sum_attributions_pos
                #relative_neg_attributions = attributions_magnitude_neg / sum_attributions_neg
                
                #all_weighted_pos_attributions.append(relative_pos_attributions)
                #all_weighted_neg_attributions.append(relative_neg_attributions)
    
                relative_attributions = attributions_magnitude/sum_attributions
    
                all_weighted_attributions.append(relative_attributions)
        
                # Amplituden speichern
                amplitudes = np.abs(x_segment_dct).cpu().detach().numpy()
                all_amplitudes.append(amplitudes)
    
    # Mittelwerte der Attributionswerte über alle Segmente hinweg für jede Frequenz berechnen (sollte eigentlich auch gewichtet sein, unterscheidet sich aber hier nur in einem Faktor)
    #mean_weighted_pos_attributions = np.mean(all_weighted_pos_attributions, axis=0)
    #mean_weighted_neg_attributions = np.mean(all_weighted_neg_attributions, axis=0)
    #mean_weighted_attributions = mean_weighted_pos_attributions + mean_weighted_neg_attributions
    mean_attributions = np.mean(all_weighted_attributions, axis=0)

    mean_weighted_pos_attributions = np.where(mean_attributions > 0, mean_attributions, 0)
    mean_weighted_neg_attributions = np.where(mean_attributions < 0, mean_attributions, 0)

    # Mittelwerte der Amplituden über alle Segmente hinweg für jede Frequenz berechnen
    mean_amplitudes = np.mean(all_amplitudes, axis=0)
    
    # Gewichteter Durchschnitt der Frequenzen berechnen
    overall_average_frequency_pos = np.sum(all_frequencies * mean_weighted_pos_attributions) / np.sum(mean_weighted_pos_attributions)
    overall_average_frequency_neg = np.sum(all_frequencies * mean_weighted_neg_attributions) / np.sum(mean_weighted_neg_attributions)

    # Speichern der durchschnittlichen Frequenzen
    overall_average_frequencies_pos_list.append(overall_average_frequency_pos)
    overall_average_frequencies_neg_list.append(overall_average_frequency_neg)
    mean_attributions_list.append(mean_attributions)
    
    # Plot 1: Average Amplitudes colored by weighted attributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # First Plot - Average Amplitudes
    bars = ax1.bar(
        all_frequencies,
        mean_amplitudes,
        color=cmap(((mean_attributions / np.max(np.abs(mean_attributions)))+1)/2),
        label="coloured by weighted attributions",
        width=0.2
    )
    
    # Settings for the first plot
    ax1.set_xlabel("Frequency (Hz)", fontsize=20)
    ax1.set_ylabel("Average Amplitudes", fontsize=20)
    ax1.set_title(f"Average weighted attributions of frequencies in {phases[i]} stage", fontsize=22)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlim(0, 40)  # Limit to 40 Hz
    ax1.set_ylim(0, np.max(mean_amplitudes) * 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Average Weighted Positive and Negative Attributions
    
    # Positive weighted attributions
    bars_pos = ax2.bar(
        all_frequencies,
        mean_weighted_pos_attributions,
        color=cmap_pos(mean_weighted_pos_attributions / np.max(np.abs(mean_attributions))),
        label="Positively weighted attributions",
        width=0.2
    )
        
    # Negative weighted attributions (negated for visual clarity)
    bars_neg = ax2.bar(
        all_frequencies,
        mean_weighted_neg_attributions,
        color=cmap_neg(-mean_weighted_neg_attributions / np.max(np.abs(mean_attributions))),
        label="Negatively weighted attributions",
        width=0.2
    )
    
    # Settings for the second plot
    ax2.set_xlabel("Frequency (Hz)", fontsize=20)
    ax2.set_ylabel("Average Attribution", fontsize=20)
    ax2.set_title(f"Average weighted positive/negative attributions in {phases[i]} stage", fontsize=22)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_xlim(0, 40)
    ax2.set_ylim(np.min(mean_weighted_neg_attributions) * 1.1, np.max(mean_weighted_pos_attributions) * 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the figure with both plots stacked vertically
    plt.tight_layout()
    pdf_file.savefig()
    plt.show()

    # Dateiname erstellen
    mat_filename = f"mean_attributions_mean_of_surrounding_segments{phases[i]}.mat"
    mat_filepath = os.path.join(output_folder, mat_filename)
    
    # Speichere die Liste als Matrix
    savemat(mat_filepath, {"mean_attributions": mean_attributions})

# Now let's add the printed output to the PDF
fig_text = plt.figure(figsize=(8, 6))  # Create a new figure for text
plt.axis('off')  # Turn off the axis

# Prepare the text output
text_output = "\nAverage (positiv weighted) frequencies for all sleep stages:\n"
for i, avg_freq in enumerate(overall_average_frequencies_pos_list):
    text_output += f"{phases[i]}: {avg_freq:.2f} Hz\n"

text_output += "\nAverage (negativ weighted) frequencies for all sleep stages:\n"
for i, avg_freq in enumerate(overall_average_frequencies_neg_list):
    text_output += f"{phases[i]}: {avg_freq:.2f} Hz\n"

text_output += "\nMost dominant (positive or negative attributed) frequencies compared to the other sleep stages:\n"
sum_mean_attributions_list = np.sum(np.abs(mean_attributions_list))
for i, attr in enumerate(mean_attributions_list):
    attr_arg = np.argmax(attr/sum_mean_attributions_list)
    frequency = all_frequencies[np.argmax(attr/sum_mean_attributions_list)]
    text_output += f"{phases[i]}: {frequency:.2f} Hz at (relative) attribution {attr[attr_arg]:.2f}\n"

# Add the text to the figure
plt.text(0.5, 0.5, text_output, fontsize=12, ha='center', va='center')

# Save the text figure to the PDF
pdf_file.savefig(fig_text)

# Close the PDF file
pdf_file.close()
