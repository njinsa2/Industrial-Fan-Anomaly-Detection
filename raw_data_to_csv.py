import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from mel_spectrogram import mel_representation


#configuration constants
MIMII_SNR_DIRS = {
    "0dB": "0_dB_fan",
    "6dB": "6_dB_fan",
    "-6dB": "-6_dB_fan"
}
OUTPUT_DIR = "data_csv"             #destination folder for generated CSV files
TARGET_SR = 16000                   #target sample rate matching dataset specifications
N_MELS = 128                        #number of mel bands
USE_DESCRIPTORS = True              #use "rich" descriptors (static + deltas + global stats)
TEST_RATIO = 0.20                   #chronological test split ratio per machine/class
PURGE_BUFFER_FILES = 20             #boundary files discarded to eliminate temporal leakage (~200 seconds)


def parse_mimii_path(audio_path: Path) -> tuple[int | None, str | None]:
    """
    Parses file path to extract:
    - target: 0 (normal) or 1 (abnormal)
    - source_id: Machine ID ('id_00', 'id_02', 'id_04', 'id_06')
    """
    parts = [p.lower() for p in audio_path.parts]

    if "normal" in parts:
        target = 0
    elif "abnormal" in parts:
        target = 1
    else:
        return None, None

    source_id = None
    for part in parts:
        if re.match(r"^id_\d{2}$", part):
            source_id = part
            break

    return target, source_id


def process_mimii_dataset():
    mode_str = "Acoustic Descriptors (Deltas + Stats)" if USE_DESCRIPTORS else "Flattened Spectrogram"
    print(f"Processing MIMII Multi-SNR Dataset | Mode: {mode_str}")
    
    rows = []
    skipped_count = 0

    #gather and extract features across all three SNR folders
    for snr_label, dir_name in MIMII_SNR_DIRS.items():
        root_path = Path(dir_name)
        if not root_path.exists():
            print(f"Warning: Directory '{dir_name}' ({snr_label}) not found. Skipping...")
            continue

        wav_paths = sorted(list(root_path.rglob("*.wav")))
        print(f"\nProcessing {len(wav_paths)} files from '{dir_name}' ({snr_label})...")

        for idx, audio_path in enumerate(wav_paths, 1):
            target, source_id = parse_mimii_path(audio_path)

            if target is None or source_id is None:
                skipped_count += 1
                continue

            try:
                feat_vector = mel_representation(
                    file_path=audio_path,
                    n_mels=N_MELS,
                    sr=TARGET_SR,
                    vector_form=True,
                    descriptors=USE_DESCRIPTORS
                )

                feature_dict = {}

                if USE_DESCRIPTORS:
                    #unpack mel band stats & temporal derivatives (5 * N_MELS)
                    for i in range(N_MELS):
                        feature_dict[f"mel_mean_{i}"] = float(feat_vector[i])
                        feature_dict[f"mel_std_{i}"] = float(feat_vector[N_MELS + i])
                        feature_dict[f"mel_max_{i}"] = float(feat_vector[2 * N_MELS + i])
                        feature_dict[f"delta_mean_{i}"] = float(feat_vector[3 * N_MELS + i])
                        feature_dict[f"delta2_mean_{i}"] = float(feat_vector[4 * N_MELS + i])

                    #unpack global bioacoustic instability descriptor statistics (10 features)
                    offset = 5 * N_MELS
                    global_names = [
                        "zcr_mean", "zcr_std", "rms_mean", "rms_std",
                        "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
                        "flatness_mean", "flatness_std"
                    ]
                    for idx_g, name in enumerate(global_names):
                        feature_dict[name] = float(feat_vector[offset + idx_g])
                else:
                    feature_dict = {f"mel_flat_{i}": float(v) for i, v in enumerate(feat_vector)}

                feature_dict["target"] = target
                feature_dict["source_id"] = source_id
                feature_dict["file_name"] = audio_path.name  #e.g., '00000015.wav'
                feature_dict["snr"] = snr_label

                rows.append(feature_dict)

            except Exception as e:
                print(f"Error processing {audio_path.name}: {e}")
                skipped_count += 1
                continue

            if idx % 1000 == 0 or idx == len(wav_paths):
                print(f"Processed {idx}/{len(wav_paths)} files in {snr_label}...")

    df_full = pd.DataFrame(rows)
    print(f"Extracted {df_full.shape[1] - 4} features for {len(df_full)} total audio files across all SNRs\n")

    #base filename chronological block split (leak-free split across snrs)
    train_0db_list = []
    test_0db_list = []
    test_6db_list = []
    test_minus6db_list = []
    total_purged_events = 0

    print("Applying Leak-Free Base Filename Chronological Block Split\n")
    for (source_id, target), group in df_full.groupby(["source_id", "target"], sort=False):
        unique_files = sorted(group["file_name"].unique())
        n_unique = len(unique_files)

        test_size = int(n_unique * TEST_RATIO)
        train_end_idx = n_unique - test_size - PURGE_BUFFER_FILES
        test_start_idx = n_unique - test_size

        if train_end_idx <= 0:
            raise ValueError(f"Purge buffer of {PURGE_BUFFER_FILES} is too large for group {source_id}-{target}.")

        train_file_set = set(unique_files[:train_end_idx])
        test_file_set = set(unique_files[test_start_idx:])
        purged_file_count = test_start_idx - train_end_idx
        total_purged_events += purged_file_count

        #compute relative time mapping for machine interleaving
        file_to_rel_time = {f_name: i / (n_unique - 1) for i, f_name in enumerate(unique_files)}

        #split by snr and time window
        sub_0db = group[group["snr"] == "0dB"]
        sub_6db = group[group["snr"] == "6dB"]
        sub_minus6db = group[group["snr"] == "-6dB"]

        tr_0db = sub_0db[sub_0db["file_name"].isin(train_file_set)].copy()
        te_0db = sub_0db[sub_0db["file_name"].isin(test_file_set)].copy()
        te_6db = sub_6db[sub_6db["file_name"].isin(test_file_set)].copy()
        te_minus6db = sub_minus6db[sub_minus6db["file_name"].isin(test_file_set)].copy()

        #map relative timestamps
        tr_0db["rel_time"] = tr_0db["file_name"].map(file_to_rel_time)
        te_0db["rel_time"] = te_0db["file_name"].map(file_to_rel_time)
        te_6db["rel_time"] = te_6db["file_name"].map(file_to_rel_time)
        te_minus6db["rel_time"] = te_minus6db["file_name"].map(file_to_rel_time)

        train_0db_list.append(tr_0db)
        test_0db_list.append(te_0db)
        test_6db_list.append(te_6db)
        test_minus6db_list.append(te_minus6db)

        label_str = "normal" if target == 0 else "abnormal"
        print(f"  - [{source_id} | {label_str:8s}]: Unique Files={n_unique:4d} | Train (0dB)={len(tr_0db):4d} | Purged={purged_file_count:2d} | Test (0dB/6dB/-6dB)={len(te_0db):4d}")

    #combine split lists
    train_0db_df = pd.concat(train_0db_list, ignore_index=True)
    test_0db_df = pd.concat(test_0db_list, ignore_index=True)
    test_6db_df = pd.concat(test_6db_list, ignore_index=True)
    test_minus6db_df = pd.concat(test_minus6db_list, ignore_index=True)

    #interleave machine ids chronologically by relative time
    train_0db_df = train_0db_df.sort_values("rel_time").reset_index(drop=True)
    test_0db_df = test_0db_df.sort_values("rel_time").reset_index(drop=True)
    test_6db_df = test_6db_df.sort_values("rel_time").reset_index(drop=True)
    test_minus6db_df = test_minus6db_df.sort_values("rel_time").reset_index(drop=True)

    #drop tracking columns right before saving
    cols_to_drop = ["file_name", "snr", "rel_time"]
    train_0db_df = train_0db_df.drop(columns=cols_to_drop)
    test_0db_df = test_0db_df.drop(columns=cols_to_drop)
    test_6db_df = test_6db_df.drop(columns=cols_to_drop)
    test_minus6db_df = test_minus6db_df.drop(columns=cols_to_drop)

    #export csv files
    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_0db_path = os.path.join(train_dir, "train_0dB.csv")
    test_0db_path = os.path.join(test_dir, "test_0dB.csv")
    test_6db_path = os.path.join(test_dir, "test_6dB.csv")
    test_minus6db_path = os.path.join(test_dir, "test_minus6dB.csv")

    train_0db_df.to_csv(train_0db_path, index=False)
    test_0db_df.to_csv(test_0db_path, index=False)
    test_6db_df.to_csv(test_6db_path, index=False)
    test_minus6db_df.to_csv(test_minus6db_path, index=False)

    print("\n--- Preprocessing Complete ---")
    print(f"Total base recording events purged per SNR: {total_purged_events}")
    print(f"Saved Train CSV : {train_0db_path} (Shape: {train_0db_df.shape})")
    print(f"Saved Test CSVs :")
    print(f"  - Control (0dB) : {test_0db_path} (Shape: {test_0db_df.shape})")
    print(f"  - Cleaner (6dB) : {test_6db_path} (Shape: {test_6db_df.shape})")
    print(f"  - Noisier (-6dB): {test_minus6db_path} (Shape: {test_minus6db_df.shape})")


if __name__ == "__main__":
    process_mimii_dataset()