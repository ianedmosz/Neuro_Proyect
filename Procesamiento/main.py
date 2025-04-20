import os
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

# 1. Configuración inicial
path = "Datos/Subjects/S2"
fs = 256  # Frecuencia de muestreo
window_duration = 5  # Duración de ventana en segundos
nperseg = fs * window_duration  # Muestras por ventana
baseline_duration = 30  # Línea base: primeros 30 segundos

# 2. Cargar y procesar datos
all_data = pd.DataFrame()

for idx, filename in enumerate(os.listdir(path)):
    if filename.endswith(".set"):
        raw = mne.io.read_raw_eeglab(os.path.join(path, filename), preload=True)
        channels = raw.ch_names[:32]
        data, _ = raw[channels, :]
        
        df = pd.DataFrame(data.T, columns=channels)
        df['ID'] = idx + 1
        all_data = pd.concat([all_data, df], ignore_index=True)

# 3. Procesar sujeto 2
df_id2 = all_data[all_data['ID'] == 1].copy()
nombres_canales = [col for col in df_id2.columns if col != 'ID']
data = df_id2[nombres_canales].values.T  # shape: (n_canales, n_muestras)

# 4. Definir regiones cerebrales
frontal_theta = ['Fz', 'F3', 'F4']
parietal_alpha = ['Pz', 'P3', 'P4']
prefrontal_beta = ['FP1', 'FP2']

# 5. Calcular línea base (primeros 30 segundos)
baseline_samples = baseline_duration * fs
freqs_baseline, psd_baseline = welch(
    data[:, :baseline_samples], 
    fs=fs, 
    nperseg=nperseg,
    axis=1
)

# Obtener índices de canal
idx_frontal = [nombres_canales.index(ch) for ch in frontal_theta if ch in nombres_canales]
idx_parietal = [nombres_canales.index(ch) for ch in parietal_alpha if ch in nombres_canales]
idx_prefrontal = [nombres_canales.index(ch) for ch in prefrontal_beta if ch in nombres_canales]

# Obtener índices de frecuencia para línea base
idx_theta_base = np.where((freqs_baseline >= 4) & (freqs_baseline <= 8))[0]
idx_alpha_base = np.where((freqs_baseline >= 8) & (freqs_baseline <= 12))[0]
idx_beta_base = np.where((freqs_baseline >= 12) & (freqs_baseline <= 30))[0]

# Calcular potencias base
#theta_baseline = psd_baseline[idx_frontal][:, idx_theta_base].mean()
#alpha_baseline = psd_baseline[idx_parietal][:, idx_alpha_base].mean()
#beta_baseline = psd_baseline[idx_prefrontal][:, idx_beta_base].mean()

# Usa percentil 75 para reducir impacto de outliers
theta_baseline = np.percentile(psd_baseline[idx_frontal][:, idx_theta_base], 75)
alpha_baseline = np.percentile(psd_baseline[idx_parietal][:, idx_alpha_base], 75)
beta_baseline = np.percentile(psd_baseline[idx_prefrontal][:, idx_beta_base], 75)

# 6. Ventanas y normalización
engagement = []
n_muestras = data.shape[1]
n_ventanas = int(np.ceil(n_muestras / nperseg))

for i in range(n_ventanas):
    start = i * nperseg
    end = min((i + 1) * nperseg, n_muestras)
    segmento = data[:, start:end]

    if segmento.shape[1] < fs:  # Mínimo 1 segundo para análisis
        continue

    freqs, psd = welch(segmento, fs=fs, nperseg=min(nperseg, segmento.shape[1]), axis=1)

    # Índices por ventana
    idx_theta = np.where((freqs >= 4) & (freqs <= 7))[0]
    idx_alpha = np.where((freqs >= 8) & (freqs <= 12))[0]
    idx_beta = np.where((freqs >= 12) & (freqs <= 30))[0]

    if len(idx_theta) == 0 or len(idx_alpha) == 0 or len(idx_beta) == 0:
        continue

    # Normalización por línea base
    theta_norm = psd[idx_frontal][:, idx_theta].mean() / theta_baseline
    alpha_norm = psd[idx_parietal][:, idx_alpha].mean() / alpha_baseline
    beta_norm = psd[idx_prefrontal][:, idx_beta].mean() / beta_baseline

    # Índices de carga cognitiva
    eng = theta_norm / alpha_norm if alpha_norm > 0 else np.nan
    eng_beta = beta_norm / (alpha_norm + theta_norm) if (alpha_norm + theta_norm) > 0 else np.nan
    fatigue_index = alpha_norm / theta_norm if theta_norm > 0 else np.nan

    engagement.append({
        'ventana': i + 1,
        'inicio_seg': start/fs,
        'fin_seg': end/fs,
        'theta_frontal_norm': theta_norm,
        'alpha_parietal_norm': alpha_norm,
        'beta_prefrontal_norm': beta_norm,
        'engagement_Index': eng_beta,
        'task_load_index': eng,
        'fatigue_index': fatigue_index
    })

# 7. Resultados finales
df_engagement = pd.DataFrame(engagement)
print(f"\nTotal de ventanas: {len(df_engagement)}")
print(df_engagement.head(10))
print(f"\nPromedio TLI normalizado: {df_engagement['task_load_index'].mean():.2f}")
print(f"Promedio EBI normalizado: {df_engagement['engagement_Index'].mean():.2f}")

# 8. Guardar a CSV
df_engagement.to_csv("engagement_normalizado.csv", index=False)


