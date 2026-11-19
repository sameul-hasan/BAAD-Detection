# BAAD Keyword Spotting — Experiment & Pipeline Log

Chronological record of environment checks, data preprocessing, speaker splits, synthetic continuous streams, baselines, student training, threshold tuning, evaluation, ONNX/INT8 export, and live/file verification.

---

## [2026-11-19T12:45:49+06:00] `env_audit` — Environment & hardware audit completed

| Metric | Value |
|---|---|
| `python` | `3.10.12` |
| `os` | `Linux-6.8.0-149-generic-x86_64-with-glibc2.35` |
| `cpu_threads` | `24` |
| `torch` | `2.5.1+cu124` |
| `torchaudio` | `2.5.1+cu124` |
| `cuda_available` | `False` |
| `gpu_hardware` | `NVIDIA GeForce GTX 1660 Ti (TU116, nouveau loaded; driver not active)` |
| `onnx` | `1.23.2` |
| `onnxruntime` | `1.23.2` |
| `quantization_ready` | `True` |

- Package `baad` linked editable via `baad_kws.pth`.
- Matplotlib configured with Bangla fonts (`Lohit Bengali`, `Mukti`).
- PortAudio shared library missing (`sounddevice` guarded with `pw-record`/`parecord`/`arecord` fallback planned).

---

## [2026-11-19T12:49:00+06:00] `build_manifest` — Scanned BAAD dataset, harmonized duplicate/overlapping labels, and wrote manifest.csv + labels.json

| Metric | Value |
|---|---|
| `total_clips` | `6086` |
| `abusive_clips` | `5285` |
| `close_clips` | `801` |
| `abusive_words` | `112` |
| `close_words` | `40` |
| `total_classes_with_special` | `154` |
| `speakers_total` | `83` |
| `harmonized_cross_subset_clips` | `23` |
| `raw_sample_rates` | `{"44100": 4472, "16000": 1310, "48000": 289, "22050": 15}` |

- Merged duplicate folders: `{'abusive:বলদ': ['49.বলদ', '12. বলদ'], 'abusive:বাল পাকনা': ['21.বাল পাকনা', '90.বাল পাকনা'], 'abusive:মাদারি': ['53.মাদারি', '86.মাদারি']}`
- Harmonized close->abusive overlapping words: `['বলদ']` (23 clips)

```text
         clips  words  speakers  minutes
subset                                  
abusive   5285    112        83    64.16
close      801     40        23     9.29
```

---

## [2026-11-19T12:49:10+06:00] `preprocess` — Resampled to 16 kHz mono, trimmed silence, peak-normalised

| Metric | Value |
|---|---|
| `clips_processed` | `6086` |
| `target_sr` | `16000` |
| `before_trim_min` | `73.4500` |
| `after_trim_min` | `73.3600` |
| `median_dur_s` | `0.6850` |
| `max_dur_s` | `3.3270` |
| `pct_gt_1_0s` | `10.7000` |
| `pct_gt_1_25s` | `2.6500` |

---

## [2026-11-19T12:49:11+06:00] `splits` — Created speaker-independent train/val/test splits

| Metric | Value |
|---|---|
| `seed` | `1337` |
| `train_clips` | `4097` |
| `val_clips` | `999` |
| `test_clips` | `990` |
| `train_speakers` | `59` |
| `val_speakers` | `12` |
| `test_speakers` | `12` |
| `missing_labels_train` | `0` |
| `missing_labels_val` | `0` |
| `missing_labels_test` | `0` |

```text
                    clips  speakers  female_ratio
speaker_pool split                               
abusive      test     871         9         0.295
             train   3514        42         0.336
             val      877         9         0.304
close        test     119         3         0.345
             train    583        17         0.307
             val      122         3         0.328
```

---

## [2026-11-19T12:51:26+06:00] `download_fleurs_fetch` — Downloaded FLEURS Bengali (bn_in) archives

| Metric | Value |
|---|---|
| `train_tsv_mb` | `3.4810` |
| `train_tar_mb` | `2034.71` |
| `dev_tsv_mb` | `0.4660` |
| `dev_tar_mb` | `278.67` |
| `test_tsv_mb` | `1.0890` |
| `test_tar_mb` | `659.70` |

---

## [2026-11-19T12:52:23+06:00] `download_fleurs_convert` — Extracted and resampled FLEURS bn_in to 16 kHz mono

| Metric | Value |
|---|---|
| `train_files` | `3006` |
| `train_hours` | `10.7260` |
| `val_files` | `402` |
| `val_hours` | `1.4480` |
| `test_files` | `920` |
| `test_hours` | `3.4330` |

---

## [2026-11-19T12:53:03+06:00] `synth_stream` — Generated deterministic val and test synthetic continuous streams

| Metric | Value |
|---|---|
| `val` | `{"streams": 250, "total_hours": 0.694, "clean_neg_hours": 0.275, "abusive_events": 220, "close_events": 86}` |
| `test` | `{"streams": 250, "total_hours": 0.691, "clean_neg_hours": 0.273, "abusive_events": 233, "close_events": 86}` |

---

## [2026-11-19T12:53:33+06:00] `baseline_mfcc` — Evaluated isolated-word MFCC + LogisticRegression baseline (unseen speakers)

| Metric | Value |
|---|---|
| `fit_sec` | `27.5500` |
| `val_word_acc` | `0.2543` |
| `val_macro_f1` | `0.2358` |
| `val_abusive_prec` | `0.9266` |
| `val_abusive_rec` | `0.9318` |
| `val_abusive_f1` | `0.9292` |
| `test_word_acc` | `0.2182` |
| `test_macro_f1` | `0.2313` |
| `test_abusive_prec` | `0.9297` |
| `test_abusive_rec` | `0.8924` |
| `test_abusive_f1` | `0.9107` |

---

## [2026-11-19T12:53:59+06:00] `train_bcresnet8_v1_smoke` — Smoke BC-ResNet-8 training completed

| Metric | Value |
|---|---|
| `run_name` | `bcresnet8_v1_smoke` |
| `smoke` | `True` |
| `device` | `cpu` |
| `params` | `256538` |
| `epochs` | `2` |
| `total_minutes` | `0.3100` |
| `epoch` | `2` |
| `train_loss` | `4.8689` |
| `val_loss` | `5.1397` |
| `val_acc` | `0.0385` |
| `val_macro_f1` | `0.0005` |
| `val_ab_prec` | `0.0000` |
| `val_ab_rec` | `0.0000` |
| `val_ab_f1` | `0.0000` |
| `val_close_fp` | `0.0000` |
| `val_unk_fp` | `0.0000` |
| `iso_acc` | `0.0000` |
| `iso_macro_f1` | `0.0000` |
| `iso_ab_f1` | `0.0000` |
| `score` | `0.0002` |
| `lr` | `0.0030` |
| `epoch_sec` | `8.3800` |

- Checkpoint: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_v1_smoke/best.pt`
- Per-epoch metrics: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_v1_smoke/metrics.csv`
- Curves: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_v1_smoke/training_curves.png`

---

## [2026-11-19T12:58:45+06:00] `train_bench_ep1` — Full BC-ResNet-6 training completed

| Metric | Value |
|---|---|
| `run_name` | `bench_ep1` |
| `smoke` | `False` |
| `device` | `cpu` |
| `params` | `151738` |
| `epochs` | `1` |
| `total_minutes` | `0.8000` |
| `epoch` | `1` |
| `train_loss` | `4.1396` |
| `val_loss` | `4.6922` |
| `val_acc` | `0.2202` |
| `val_macro_f1` | `0.0076` |
| `val_ab_prec` | `0.0000` |
| `val_ab_rec` | `0.0000` |
| `val_ab_f1` | `0.0000` |
| `val_close_fp` | `0.0000` |
| `val_unk_fp` | `0.0000` |
| `iso_acc` | `0.0000` |
| `iso_macro_f1` | `0.0000` |
| `iso_ab_f1` | `0.0000` |
| `score` | `0.0038` |
| `lr` | `0.0030` |
| `epoch_sec` | `47.5200` |

- Checkpoint: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bench_ep1/best.pt`
- Per-epoch metrics: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bench_ep1/metrics.csv`
- Curves: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bench_ep1/training_curves.png`

---

## [2026-11-19T12:59:01+06:00] `cpu_tuning_profile` — Benchmarked CPU DataLoader & OpenMP thread scaling; selected fast CPU config

| Metric | Value |
|---|---|
| `threads_24_scale8_step_s` | `1.6010` |
| `threads_6_scale8_step_s` | `0.6850` |
| `threads_12_scale6_step_s` | `0.3940` |
| `threads_8_scale6_epoch_s` | `47.5200` |
| `selected_scale` | `6` |
| `selected_params` | `151738` |
| `selected_samples_per_epoch` | `6400` |
| `selected_epochs` | `30` |
| `num_workers_cpu` | `0` |

- Forking DataLoader workers with OpenMP caused severe thread contention; `num_workers=0` + `torch.set_num_threads(8)` eliminated the stall (dataset slicing = 0.17s / 256 samples).
- `BCResNet(scale=6)` runs ~47.5s/epoch (~24 min total for 30 epochs) and yields an even smaller mobile model.

---

## [2026-11-19T13:00:50+06:00] `gpu_pause` — Stopped CPU training per user request; waiting for NVIDIA driver install + reboot for GPU-only training

| Metric | Value |
|---|---|
| `cuda_available` | `False` |
| `gpu_hardware` | `NVIDIA GeForce GTX 1660 Ti (TU116, currently on nouveau)` |
| `data_pipeline_complete` | `True` |
| `fleurs_hours` | `15.6100` |
| `synth_val_test_streams` | `500` |
| `unit_tests_passing` | `4/4 (manifest, splits_no_leak, synth_labels, detector)` |
| `next_gpu_config` | `bcresnet8_gpu_v1 (scale=8, batch_size=128, epochs=50, fp16 AMP)` |

- Cancelled CPU training `bcresnet6_v1` (`task-232`).
- Updated `train_student.py` to enforce `--device cuda` by default (raises error instead of silent CPU fallback) and use FP16 AMP (`torch.amp`).
- All preprocessed data, splits, FLEURS-bn audio, and synthetic streams are saved under `baad_kws/data/`.

---

## [2026-11-19T13:24:36+06:00] `env_gpu_ready` — NVIDIA GTX 1660 Ti driver active and verified inside sandbox

| Metric | Value |
|---|---|
| `cuda_available` | `True` |
| `gpu_name` | `NVIDIA GeForce GTX 1660 Ti` |
| `vram_mb` | `5747.40` |
| `driver_cuda` | `570.211.01 / CUDA 12.8` |
| `torch` | `2.5.1+cu124` |

---

## [2026-11-19T13:24:50+06:00] `train_bcresnet8_gpu_smoke` — Smoke BC-ResNet-8 training completed

| Metric | Value |
|---|---|
| `run_name` | `bcresnet8_gpu_smoke` |
| `smoke` | `True` |
| `device` | `cuda` |
| `gpu_name` | `NVIDIA GeForce GTX 1660 Ti` |
| `peak_vram_mb` | `923.80` |
| `params` | `256538` |
| `epochs` | `2` |
| `total_minutes` | `0.1000` |
| `epoch` | `1` |
| `train_loss` | `5.4635` |
| `val_loss` | `5.2603` |
| `val_acc` | `0.0038` |
| `val_macro_f1` | `0.0000` |
| `val_ab_prec` | `0.6774` |
| `val_ab_rec` | `1.0000` |
| `val_ab_f1` | `0.8077` |
| `val_close_fp` | `1.0000` |
| `val_unk_fp` | `1.0000` |
| `iso_acc` | `0.0050` |
| `iso_macro_f1` | `0.0001` |
| `iso_ab_f1` | `0.9367` |
| `score` | `0.4039` |
| `lr` | `0.0030` |
| `epoch_sec` | `2.6800` |

- Checkpoint: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_gpu_smoke/best.pt`
- Per-epoch metrics: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_gpu_smoke/metrics.csv`
- Curves: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_gpu_smoke/training_curves.png`

---

## [2026-11-19T13:47:39+06:00] `train_bcresnet8_gpu_v1` — Full BC-ResNet-8 training completed

| Metric | Value |
|---|---|
| `run_name` | `bcresnet8_gpu_v1` |
| `smoke` | `False` |
| `device` | `cuda` |
| `gpu_name` | `NVIDIA GeForce GTX 1660 Ti` |
| `peak_vram_mb` | `923.80` |
| `params` | `256538` |
| `epochs` | `60` |
| `total_minutes` | `22.4500` |
| `epoch` | `48` |
| `train_loss` | `0.8632` |
| `val_loss` | `1.3573` |
| `val_acc` | `0.8314` |
| `val_macro_f1` | `0.8341` |
| `val_ab_prec` | `0.9864` |
| `val_ab_rec` | `0.8273` |
| `val_ab_f1` | `0.8999` |
| `val_close_fp` | `0.0252` |
| `val_unk_fp` | `0.0280` |
| `iso_acc` | `0.8288` |
| `iso_macro_f1` | `0.8653` |
| `iso_ab_f1` | `0.9219` |
| `score` | `0.8670` |
| `lr` | `0.0004` |
| `epoch_sec` | `22.4800` |

- Checkpoint: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_gpu_v1/best.pt`
- Per-epoch metrics: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_gpu_v1/metrics.csv`
- Curves: `/home/sameul4740/Sameul/BAAD/baad_kws/runs/bcresnet8_gpu_v1/training_curves.png`

---

## [2026-11-19T13:51:27+06:00] `evaluate_bcresnet8_gpu_v1` — Evaluated best.pt on isolated test, val tuning, test continuous streams, and Raw Data

| Metric | Value |
|---|---|
| `val_iso_top1` | `0.8288` |
| `val_iso_top5` | `0.9550` |
| `val_iso_macro_f1` | `0.8653` |
| `val_iso_ab_prec` | `0.9960` |
| `val_iso_ab_rec` | `0.8580` |
| `val_iso_ab_f1` | `0.9219` |
| `test_iso_top1` | `0.8707` |
| `test_iso_top5` | `0.9859` |
| `test_iso_macro_f1` | `0.8722` |
| `test_iso_ab_prec` | `0.9889` |
| `test_iso_ab_rec` | `0.9188` |
| `test_iso_ab_f1` | `0.9526` |
| `tuned_smooth_k` | `4` |
| `tuned_threshold` | `0.8500` |
| `tuned_binary_threshold` | `0.9500` |
| `val_event_any_f1` | `0.7430` |
| `val_fa_per_hour` | `14.5300` |
| `test_event_any_prec` | `0.9351` |
| `test_event_any_rec` | `0.6180` |
| `test_event_any_f1` | `0.7442` |
| `test_event_exact_prec` | `0.8961` |
| `test_event_exact_rec` | `0.5923` |
| `test_event_exact_f1` | `0.7132` |
| `test_sent_prec` | `0.9565` |
| `test_sent_rec` | `0.7333` |
| `test_sent_f1` | `0.8302` |
| `test_sent_auc` | `0.9213` |
| `test_clean_neg_fa` | `5` |
| `test_clean_neg_hours` | `0.2727` |
| `test_fa_per_hour` | `18.3400` |
| `test_close_neg_fa_rate` | `0.1000` |
| `raw_test_files` | `9` |
| `raw_total_detections` | `522` |

- Tuned config: `runs/bcresnet8_gpu_v1/eval/detect_tuned.yaml`
- DET plot: `runs/bcresnet8_gpu_v1/eval/det_curve.png`
- Errors CSV: `runs/bcresnet8_gpu_v1/eval/test_errors.csv`

---

## [2026-11-19T13:51:54+06:00] `export_bcresnet8_gpu_v1` — Exported FP32 (1441.4 KB) and INT8 (843.6 KB) ONNX models

| Metric | Value |
|---|---|
| `fp32_kb` | `1441.40` |
| `int8_kb` | `843.60` |
| `int8_mb` | `0.8240` |
| `pt_vs_fp32_max_abs_diff` | `0.0000` |
| `fp32_vs_int8_top1_agree` | `0.9150` |
| `fp32_vs_int8_cos_sim` | `0.9950` |
| `t1_mean_ms` | `2.7640` |
| `t1_median_ms` | `2.7610` |
| `t1_p95_ms` | `2.7850` |
| `t2_mean_ms` | `2.2870` |
| `t2_median_ms` | `2.2870` |
| `t2_p95_ms` | `2.3050` |

- Export bundle: `runs/bcresnet8_gpu_v1/export` (`baad_kws_int8.onnx`, `baad_kws_fp32.onnx`, `labels.json`, `detect_tuned.yaml`)

---

## [2026-11-19T13:54:15+06:00] `evaluate_bcresnet8_gpu_v1` — Evaluated best.pt on isolated test, val tuning, test continuous streams, and Raw Data

| Metric | Value |
|---|---|
| `val_iso_top1` | `0.8288` |
| `val_iso_top5` | `0.9550` |
| `val_iso_macro_f1` | `0.8653` |
| `val_iso_ab_prec` | `0.9960` |
| `val_iso_ab_rec` | `0.8580` |
| `val_iso_ab_f1` | `0.9219` |
| `test_iso_top1` | `0.8707` |
| `test_iso_top5` | `0.9859` |
| `test_iso_macro_f1` | `0.8722` |
| `test_iso_ab_prec` | `0.9889` |
| `test_iso_ab_rec` | `0.9188` |
| `test_iso_ab_f1` | `0.9526` |
| `tuned_smooth_k` | `4` |
| `tuned_threshold` | `0.9000` |
| `tuned_binary_threshold` | `0.9500` |
| `val_event_any_f1` | `0.7314` |
| `val_fa_per_hour` | `3.6300` |
| `test_event_any_prec` | `0.9632` |
| `test_event_any_rec` | `0.5622` |
| `test_event_any_f1` | `0.7100` |
| `test_event_exact_prec` | `0.9485` |
| `test_event_exact_rec` | `0.5536` |
| `test_event_exact_f1` | `0.6992` |
| `test_sent_prec` | `0.9712` |
| `test_sent_rec` | `0.6733` |
| `test_sent_f1` | `0.7953` |
| `test_sent_auc` | `0.9213` |
| `test_clean_neg_fa` | `3` |
| `test_clean_neg_hours` | `0.2727` |
| `test_fa_per_hour` | `11.0000` |
| `test_close_neg_fa_rate` | `0.0667` |

- Tuned config: `runs/bcresnet8_gpu_v1/eval/detect_tuned.yaml`
- DET plot: `runs/bcresnet8_gpu_v1/eval/det_curve.png`
- Errors CSV: `runs/bcresnet8_gpu_v1/eval/test_errors.csv`

---

## [2026-11-19T13:55:31+06:00] `eval_int8_continuous` — Verified INT8 ONNX continuous test-stream parity

| Metric | Value |
|---|---|
| `smooth_k` | `4` |
| `threshold` | `0.9000` |
| `binary_threshold` | `0.9500` |
| `event_any_prec` | `0.9262` |
| `event_any_rec` | `0.4850` |
| `event_any_f1` | `0.6366` |
| `event_exact_prec` | `0.9016` |
| `event_exact_rec` | `0.4721` |
| `event_exact_f1` | `0.6197` |
| `sent_prec` | `0.9789` |
| `sent_rec` | `0.6200` |
| `sent_f1` | `0.7592` |
| `sent_auc` | `0.9024` |
| `clean_neg_fa` | `2` |
| `clean_neg_hours` | `0.2727` |
| `fa_per_hour` | `7.3400` |
| `close_neg_fa_rate` | `0.0333` |

---

## [2026-11-19T13:55:59+06:00] `live_demo_file` — Marked test_pos_0000.wav: offensive=True (1 events)

| Metric | Value |
|---|---|
| `duration_s` | `10.8780` |
| `num_abusive_events` | `1` |
| `vad_active_ratio` | `1.0000` |

---

## [2026-11-19T13:56:01+06:00] `live_demo_file` — Marked test_neg_pure_0030.wav: offensive=False (0 events)

| Metric | Value |
|---|---|
| `duration_s` | `9.2500` |
| `num_abusive_events` | `0` |
| `vad_active_ratio` | `1.0000` |

---

## [2026-11-19T13:56:49+06:00] `live_demo_file` — Marked test_pos_0000.wav: offensive=True (1 events)

| Metric | Value |
|---|---|
| `duration_s` | `10.8780` |
| `num_abusive_events` | `1` |
| `vad_active_ratio` | `1.0000` |

---

## [2026-11-19T13:57:16+06:00] `final_verification` — All 5 automated pytest suites passed after final edit; file-marking demo verified

| Metric | Value |
|---|---|
| `pytest_passed` | `5` |
| `pytest_failed` | `0` |
| `pytest_duration_s` | `see pytest output` |
| `demo_pos_file` | `data/synth/test/audio/test_pos_0000.wav` |
| `demo_pos_detected` | `গু-খা (0.9754) at [0.150, 0.900]` |
| `demo_neg_file` | `data/synth/test/audio/test_neg_pure_0030.wav` |
| `demo_neg_detected` | `0` |

---

