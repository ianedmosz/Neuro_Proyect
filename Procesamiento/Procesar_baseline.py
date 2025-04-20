import os
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

# Configuración inicial
path = "Datos/Subjects"
fs = 250  # Frecuencia de muestreo (ajustar según tus datos)
nperseg = fs * 2  # 2 segundos por ventana

baseline_duration=120
baseline_samples=baseline_duration*fs


# Grupos de canales (ajustar según tu configuración)
frontal_theta = ['Fz', 'F3', 'F4']
parietal_alpha = ['Pz', 'P3', 'P4']
prefrontal_beta = ['FP1', 'FP2']

all_results = []

# Iterar sobre cada sujeto
for subject_id, subject_folder in enumerate(sorted(os.listdir(path)), start=1):
    subject_path = os.path.join(path, subject_folder)

    if not os.path.isdir(subject_path):
        continue

    # Buscar archivos baseline y experimental
    baseline_file = None
    experimental_file = None

    for file in os.listdir(subject_path):
        if 'baseline' in file.lower():
            baseline_file = os.path.join(subject_path, file)
        elif 'postexperimental' in file.lower() or 'experimental' in file.lower():
            experimental_file = os.path.join(subject_path, file)

    if not baseline_file or not experimental_file:
        print(f"Archivos faltantes para el sujeto {subject_folder}")
        continue

    try:
        # Procesar Baseline
        raw_baseline = mne.io.read_raw_eeglab(baseline_file, preload=True)
        nombres_canales = raw_baseline.ch_names

        # Obtener índices de los canales
        idx_frontal = [nombres_canales.index(ch) for ch in frontal_theta if ch in nombres_canales]
        idx_parietal = [nombres_canales.index(ch) for ch in parietal_alpha if ch in nombres_canales]
        idx_prefrontal = [nombres_canales.index(ch) for ch in prefrontal_beta if ch in nombres_canales]

        # Calcular PSD para baseline
        data_baseline, _ = raw_baseline[nombres_canales, :]
        freqs_baseline, psd_baseline = welch(
            data_baseline,
            fs=fs,
            nperseg=nperseg,
            axis=1
        )

        # Calcular valores de referencia
        idx_theta_base = np.where((freqs_baseline >= 4) & (freqs_baseline <= 7))[0]
        idx_alpha_base = np.where((freqs_baseline >= 8) & (freqs_baseline <= 12))[0]
        idx_beta_base = np.where((freqs_baseline >= 12) & (freqs_baseline <= 30))[0]

        theta_baseline = psd_baseline[idx_frontal][:, idx_theta_base].mean()
        alpha_baseline = psd_baseline[idx_parietal][:, idx_alpha_base].mean()
        beta_baseline = psd_baseline[idx_prefrontal][:, idx_beta_base].mean()

        # Procesar datos experimentales
        raw_experimental = mne.io.read_raw_eeglab(experimental_file, preload=True)
        data_exp, _ = raw_experimental[nombres_canales, :]

        # Procesar ventanas
        n_muestras = data_exp.shape[1]
        n_ventanas = int(np.ceil(n_muestras / nperseg))

        for i in range(n_ventanas):
            start = i * nperseg
            end = min((i + 1) * nperseg, n_muestras)
            segmento = data_exp[:, start:end]

            if segmento.shape[1] < fs:
                continue

            freqs, psd = welch(segmento, fs=fs, nperseg=min(nperseg, segmento.shape[1]), axis=1)

            idx_theta = np.where((freqs >= 4) & (freqs <= 8))[0]
            idx_alpha = np.where((freqs >= 8) & (freqs <= 12))[0]
            idx_beta = np.where((freqs >= 12) & (freqs <= 30))[0]

            if len(idx_theta) == 0 or len(idx_alpha) == 0 or len(idx_beta) == 0:
                continue

            # Normalización usando el baseline específico del sujeto
            theta_norm = psd[idx_frontal][:, idx_theta].mean() / theta_baseline
            alpha_norm = psd[idx_parietal][:, idx_alpha].mean() / alpha_baseline
            beta_norm = psd[idx_prefrontal][:, idx_beta].mean() / beta_baseline

            # Cálculo de índices
            eng = theta_norm / alpha_norm if alpha_norm > 0 else np.nan
            eng_beta = beta_norm / (alpha_norm + theta_norm) if (alpha_norm + theta_norm) > 0 else np.nan
            fatigue_index = alpha_norm / theta_norm if theta_norm > 0 else np.nan

            all_results.append({
                'ID': subject_id,
                'Condición': 'Experimental',
                'Ventana': i + 1,
                'Inicio_seg': start/fs,
                'Fin_seg': end/fs,
                'Theta_frontal_norm': theta_norm,
                'Alpha_parietal_norm': alpha_norm,
                'Beta_prefrontal_norm': beta_norm,
                'Engagement_Index': eng_beta,
                'Task_load_index': eng,
                'Fatigue_index': fatigue_index
            })

    except Exception as e:
        print(f"Error procesando sujeto {subject_folder}: {str(e)}")
        continue

results_df = pd.DataFrame(all_results)
#display(results_df)
#Engagment index promedio
print("Emgagement promedio")
print(results_df['Engagement_Index'].mean())

#Task load index promedio
print("Task load index promedio")
print(results_df['Task_load_index'].mean())

#Fatigue index promedio
print("Fatigue index promedio")
print(results_df['Fatigue_index'].mean())

# Guardar resultados
results_df.to_csv("engagement_results.csv", index=False)
