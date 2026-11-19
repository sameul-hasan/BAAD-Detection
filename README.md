# BAAD-Detection

Lightweight, streaming Bangla abusive-word keyword spotting (`BC-ResNet-8`, **843.6 KB INT8 ONNX**, **2.29 ms** per 100 ms hop on CPU) trained on the **BAAD** dataset and **Google FLEURS Bengali (`bn_in`)**.

## Highlights

- **Vocabulary (`154` classes)**: `112` abusive Bangla words + `40` acoustically close clean words (hard negatives) + `_unknown_` + `_silence_`.
- **Speaker-independent evaluation**: Strict speaker grouping (`59` train / `12` val / `12` test speakers, zero leakage).
- **Isolated unseen-speaker accuracy (`test`, `990` clips)**:
  - MFCC + Logistic Regression baseline: `21.82%` top-1, `0.2313` macro-F1
  - **BC-ResNet-8 (FP32)**: **`87.07%` top-1**, **`98.59%` top-5**, **`0.8722` macro-F1**, **`95.26%` binary abusive F1**
  - **BC-ResNet-8 (INT8 ONNX)**: **`84.55%` top-1**, **`0.8546` macro-F1**
- **Continuous streaming detection (`test`, `250` synthetic streams)**:
  - FP32: `96.32%` event precision, `56.22%` recall, `0.7100` event F1 (`±0.5 s`), `0.9213` sentence AUC
  - INT8 ONNX: `92.62%` event precision, `48.50%` recall, `0.6366` event F1, `0.9024` sentence AUC, `7.34 FA/h`
- **Mobile-ready ONNX export**: `843.6 KB` (`0.824 MB`) `baad_kws_int8.onnx` with built-in `Conv1d` `LogMel` front-end (`win=320, n_fft=320, hop=160, n_mels=40`), `2.29 ms` median / `2.31 ms` p95 latency on 2 CPU threads.

## Project Layout

```text
BAAD/
├── README.md
├── PlanforBAAD.md
└── baad_kws/
    ├── configs/                  # Training & tuned detection configs
    ├── src/baad/
    │   ├── data/                 # Manifest, 16 kHz preprocessing, speaker splits, FLEURS, augment, synth streams
    │   ├── models/               # BC-ResNet-8 + MFCC baseline
    │   ├── features.py           # ONNX-exportable Conv1d LogMel front-end
    │   ├── train_student.py      # GPU FP16 AMP training loop
    │   ├── detector.py           # EnergyVAD + sliding window + smoothing + peak picking
    │   ├── evaluate.py           # Isolated & continuous metrics, DET tuning, Raw Data eval
    │   ├── export.py             # ONNX FP32 + static QDQ INT8 export & latency parity
    │   ├── live.py               # Live microphone & audio file visual marking demo
    │   └── logging_utils.py      # Structured JSONL & Markdown experiment logs
    ├── tests/                    # Automated test suite (5/5 passing)
    ├── logs/                     # Full chronological experiment & verification logs
    └── runs/bcresnet8_gpu_v1/    # Exported ONNX bundle, eval CSVs, plots, demo outputs
```

## Quickstart

```bash
cd baad_kws
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "$PWD/src" > .venv/lib/python3.10/site-packages/baad_kws.pth

# Run live microphone detection (visual Bangla word marking in terminal / --gui)
.venv/bin/python -m baad.live --export-dir runs/bcresnet8_gpu_v1/export --gui

# Mark an audio file and save JSON + highlighted waveform PNG
.venv/bin/python -m baad.live --export-dir runs/bcresnet8_gpu_v1/export \
  --file path/to/audio.wav --json out.json --save-plot marked.png

# Run tests
.venv/bin/pytest -v tests/
```

## Dataset Reference

- Hossain, Md. F., Supto, Md. A. A., Chowdhury, Z., Chowdhury, H. S., & Abujar, S. (2023). *BAAD: A multipurpose dataset for automatic Bangla offensive speech recognition*. Data in Brief, 48, 109067. https://doi.org/10.1016/j.dib.2023.109067
- Background Bangla carrier speech: Google FLEURS (`bn_in`).
