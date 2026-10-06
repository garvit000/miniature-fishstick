# Dataset Directory

The raw EPIStress dataset and preprocessed multimodal feature cache reside in this directory:

- `EPIStress/`: Raw participant folders (`ES140/`, `ES141/`, ..., `ES157/`) containing multi-sensor pickles and CSV records. Downloadable via `python download_all.py`.
- `features_multimodal.csv`: Extracted consolidated multimodal feature matrix across all 18 participants and 144 experimental sessions (~12,586 samples × 229 columns). Automatically constructed on the first run of `python main.py` or when passing `--force-rebuild`.

These files are gitignored due to size considerations (>270 MB combined).
