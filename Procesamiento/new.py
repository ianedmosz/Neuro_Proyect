import mne

path='Sujeto2_PostExperimental_Baseline.set'

raw=mne.io.read_raw_eeglab(path, preload=True)

print(raw.info)

raw.pick_channels(['Fz', 'Cz', 'Pz'])
raw.pick(['Fz', 'Cz', 'Pz'])
raw.plot()