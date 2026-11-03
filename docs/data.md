# Data preparation

The bundled dataset contains LAFAN1 motions retargeted to Unitree G1 with GMR. Use the included `outputs/reference/library.npz` and `outputs/prior.pt` to train without rebuilding the data.

To build your own library, export GMR motions as **native 30 Hz NPZ** files:

| Field | Format |
| --- | --- |
| `fps` | Scalar, 30 Hz |
| `joint_names`, `body_names` | Name arrays; bodies must include `pelvis` |
| `joint_pos` | `(T, J)`, radians |
| `body_pos_w` | `(T, B, 3)`, meters |
| `body_quat_w` | `(T, B, 4)`, wxyz |
| `metadata_json` | Optional JSON string with source information |

```bash
spectral-prior build --input /path/to/gmr_npz_30hz \
  --selections spectral_prior/configs/reference_selection.json
spectral-prior train --steps 5000 --device cpu
spectral-prior evaluate
```

The selection file specifies source windows using 30 Hz frame indices. Adjust it for your inputs, or omit `--selections` for automatic selection. Processing selects ten bilateral leg/arm joints, repeats each cycle to 10 seconds, resamples to 60 Hz and smooths the trajectories. Source and processing details are recorded in `outputs/reference/curation.json`.

The bundled library is independently curated, not the authors’ original dataset. Clip 1 mirrors the right arm with a half-cycle shift. Clip 4 uses clip 5’s six hip/knee/ankle pitch trajectories, phase-aligned and resampled to clip 4’s original stride period; its arms are unchanged. The speed label remains the original source-window measurement, not a measured speed of the edited gait. Both edits are reproducible through the selection file and recorded in the metadata.

Train a new policy after replacing the prior. For playback or resume, keep the run’s bundled `spectral_prior.pt` with its checkpoint and `run_config.json`.
