import os
import re
import pandas as pd

# %%
def get_participants(base_path: str) -> list[str]:
    """
    Scan base_path for participant folders name,
    and return them sorted by their numeric ID.
    """
    all_dirs = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    # extract number with regex, sort numerically
    participants = sorted(
        all_dirs,
        key=lambda n: int(re.search(r'\d+', n).group())
    )
    return participants

# %%
# 🌳 Data from situ experiments

# %%
### Load the Raw data for all Participant containing Muse EEG, Empatica ACC/BVP/EDA/TEMP, PsychoPy logs & questionnaires, stretched data

# %%
def process_situ_data_raw(
    base_path: str,
    participants: list[str],
    modalities: list[str] = ['Empatica', 'Muse', 'Questionnaire', 'Stretched']
) -> None:
    
    for p in participants:
        print(f"\n=== Participant: {p} ===")
        raw_root = os.path.join(base_path, p, 'situ', 'Raw')

        for mod in modalities:
            print(f"\n-- {mod} CSV files --")
            mod_path = os.path.join(raw_root, mod)

            if not os.path.isdir(mod_path):
                print(f"   ✗ Missing folder: {mod_path}")
                continue

            # find CSVs only
            csvs = [f for f in os.listdir(mod_path)
                    if f.lower().endswith('.csv')]
            if not csvs:
                print("   (no CSVs here)")
                continue

            for fname in csvs:
                full = os.path.join(mod_path, fname)
                print(f"  • {fname}")
                try:
                    df = pd.read_csv(full)
                    print(df.head(), "\n")
                except Exception as e:
                    print(f"    ⚠ Couldn’t read {fname}: {e}")


# %%
# Loading Labeled, Preprocessed, Features data from situ

# %%
### Labeled contains synchronized, segmented raw data per task; Preprocessed contains cleaned & preprocessed labeled data; Features contains features extracted from the preprocessed data.

# %%
def process_situ_data_labeled_preprocessed_features(
    base_path: str,
    participants: list[str],
    situ_subfolder: str
) -> None:
    
    for p in participants:
        print(f"\n=== Participant: {p} | Situ/{situ_subfolder} ===")
        root = os.path.join(base_path, p, 'Situ', situ_subfolder)

        if not os.path.isdir(root):
            print(f"   ✗ No {situ_subfolder} data for {p}")
            continue

        tasks = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
        if not tasks:
            print("   (no task folders here)")
            continue

        for task in tasks:
            task_path = os.path.join(root, task)
            print(f"\n-- {task} --")
            pickles = [
                f for f in os.listdir(task_path)
                if f.lower().endswith('.pickle')
            ]

            if not pickles:
                print("   (no .pickle files here)")
                continue

            for fname in pickles:
                full = os.path.join(task_path, fname)
                print(f"  • {fname}", end='  ')
                try:
                    # Let pandas handle the load (it remaps moved modules)
                    data = pd.read_pickle(full)
                    # Peek if DataFrame-like
                    if hasattr(data, 'head'):
                        print("\n", data.head(), "\n")
                    else:
                        print(f"(loaded: {type(data).__name__})\n")
                except Exception as e:
                    print(f"⚠ couldn’t read {fname}: {e}")


# %%
# 🧪 Data from controlled laboratory experiments

# %%
### Function to load Raw data from Lab

# %%
def process_lab_data_Raw(
    base_path: str,
    participants: list[str],
    modalities: list[str] = ['Empatica', 'Muse', 'Psychopy', 'Stretched']
) -> None:
    
    for p in participants:
        lab_root = os.path.join(base_path, p, 'Lab')
        print(f"\n=== Participant: {p} (Lab) ===")

        if not os.path.isdir(lab_root):
            print(f"   ✗ No Lab data found for participant {p}")
            continue

        raw_root = os.path.join(lab_root, 'Raw')
        for mod in modalities:
            print(f"\n-- {mod} CSV files --")
            mod_path = os.path.join(raw_root, mod)

            if not os.path.isdir(mod_path):
                print(f"   ✗ Missing folder: {mod_path}")
                continue

            csvs = [f for f in os.listdir(mod_path)
                    if f.lower().endswith('.csv')]
            if not csvs:
                print("   (no CSVs here)")
                continue

            for fname in csvs:
                full = os.path.join(mod_path, fname)
                print(f"  • {fname}")
                try:
                    df = pd.read_csv(full)
                    print(df.head(), "\n")
                except Exception as e:
                    print(f"    ⚠ Couldn’t read {fname}: {e}")

# %%
## Funtion to Load Labeled, Preprocessed, Features data from Lab

# %%
def process_lab_data_labeled_preprocessed_features(
    base_path: str,
    participants: list[str],
    lab_subfolder: str
) -> None:
    
    for p in participants:
        print(f"\n=== Participant: {p} | Lab/{lab_subfolder} ===")
        root = os.path.join(base_path, p, 'Lab', lab_subfolder)

        if not os.path.isdir(root):
            print(f"   ✗ No {lab_subfolder} data for {p}")
            continue

        tasks = [d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d))]
        if not tasks:
            print("   (no task folders here)")
            continue

        for task in tasks:
            task_path = os.path.join(root, task)
            print(f"\n-- {task} --")
            pickles = [f for f in os.listdir(task_path)
                       if f.lower().endswith('.pickle')]

            if not pickles:
                print("   (no .pickle files here)")
                continue

            for fname in pickles:
                full = os.path.join(task_path, fname)
                print(f"  • {fname}", end='  ')
                try:
                    data = pd.read_pickle(full)
                    if hasattr(data, 'head'):
                        print("\n", data.head(), "\n")
                    else:
                        print(f"(loaded: {type(data).__name__})\n")
                except Exception as e:
                    print(f"⚠ couldn’t read {fname}: {e}")

# %%
## 🖥️ Output

Run the relevent cell below to see your results.  
Make sure you’ve set the `base_path` to where your data is stored 📂  

# %%
if __name__ == "__main__":
    
    # !!!!!!!! 🚨🚨🚨🚨  Put your base path here  🚨🚨🚨🚨 
    base_path = ' '

    participants = get_participants(base_path)


# %%
## Run to see the Raw features in Situ

# %%
process_situ_data_raw(base_path, participants)

# %%
## Run to see the Preprocessed output in Situ

# %%
process_situ_data_labeled_preprocessed_features(base_path, participants, 'Preprocessed')

# %%
## Run to see the Features output in Situ

# %%
process_situ_data_labeled_preprocessed_features(base_path, participants, 'Features')

# %%
## Run to see the Labeled output in Situ

# %%
process_situ_data_labeled_preprocessed_features(base_path, participants, 'Labeled')

# %%
## Run to see the raw output in Lab

# %%
process_lab_data_Raw(base_path, participants)

# %%
## Run to see the Preprocessed output in Lab


# %%
process_lab_data_labeled_preprocessed_features(base_path, participants, 'Preprocessed')

# %%
## Run to see the Features output in Lab

# %%
process_lab_data_labeled_preprocessed_features(base_path, participants, 'Features')

# %%
## Run to see the Labeled output in Lab

# %%
process_lab_data_labeled_preprocessed_features(base_path, participants, 'Labeled')