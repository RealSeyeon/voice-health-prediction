import pandas as pd
import shutil, os
import librosa
import soundfile as sf
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Gain
from tqdm import tqdm
import numpy as np

# ✅ 폴더 정의 추가
base_dir = '/home/gpu_02/parkinson/data'
folders = ['HC_M', 'HC_F', 'PD_M', 'PD_F']


# STEP 2️⃣: 오디오 증강
original_total = 0
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    original_total += len([f for f in os.listdir(folder_path) if f.endswith('.wav')])

print(f"✅ 증강 전 총 데이터 개수: {original_total}개")

augmented_folder = os.path.join(base_dir, 'augmented')
os.makedirs(augmented_folder, exist_ok=True)

target_sr = 44100
augment = Compose([
    PitchShift(-2, 2, p=0.8),
    TimeStretch(0.9, 1.1, p=0.8),
    AddGaussianNoise(0.001, 0.015, p=0.5),
    Gain(-6, 6, p=0.5)
])

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    augmented_subfolder = os.path.join(augmented_folder, folder)
    os.makedirs(augmented_subfolder, exist_ok=True)

    for filename in tqdm([f for f in os.listdir(folder_path) if f.endswith('.wav')], desc=f"Augmenting {folder}"):
        audio, _ = librosa.load(os.path.join(folder_path, filename), sr=target_sr)
        audio = librosa.util.normalize(audio)

        sf.write(os.path.join(augmented_subfolder, f"original_{filename}"), audio, target_sr, subtype='PCM_16')

        for i in range(3):
            augmented_audio = augment(audio, sample_rate=target_sr)
            sf.write(os.path.join(augmented_subfolder, f"{filename[:-4]}_aug_{str(i+1).zfill(2)}.wav"),
                     augmented_audio, target_sr, subtype='PCM_16')

print("✅ 오디오 데이터 증강 완료!")

augmented_total = sum(len(files) for _, _, files in os.walk(augmented_folder))
print(f"✅ 증강 후 총 데이터 개수: {augmented_total}개")
