# Anisotropic EM Super-Resolution SFT

This project extends the Cosmos Framework generator SFT path with paired
electron-microscopy data. It uses the framework's Transfer batch contract:

```text
control video (synthetic thick sections) -> target video (fine EM)
```

EM depth is represented as video time. Each sample is two aligned RGB video
streams made by repeating a normalized grayscale EM crop across three channels.
The control is generated online from the target using slab integration,
nearest-neighbour section expansion, section jitter, missing sections, noise,
and contrast variation. The target itself remains clean.

The implementation uses Cosmos Framework's `video_transfer` path, therefore
the control is visible to the generator as a separate vision stream rather than
being concatenated into RGB channels.

## Step 1 — Install Cosmos Framework and this extension

Use a Cosmos Framework environment with its training dependencies installed.
Then install this repository into the same environment:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
uv sync --all-extras --group=cu130-train
source .venv/bin/activate

cd /path/to/nanoCosmosFW
pip install -e .
```

The wrapper entry point registers the `anisotropic_em_transfer_sft_edge`
experiment before it starts the framework trainer.

## Step 2 — Prepare a fine-target manifest

Write one JSON record per **registered fine EM target** in a JSONL file:

```json
{"uuid":"hela-4nm-0","target_path":"/home/nvidia/data/COSEM3D/jrc_hela-3_4nm_x4096_y0_z2048_volume.h5","target_resolution_nm":[4,4,4],"imaging":"FIB-SEM"}
```

`target_path` supports HDF5 (`.h5`/`.hdf5`), TIFF, and NPY volumes in z/y/x
order. HDF5 defaults to the first top-level dataset; add `target_key` when the
file has multiple datasets.

Supervised reconstruction requires a real fine target. A 40x4x4 nm input alone
does not contain a 4 nm z target, so it must not be used as both streams. Add
such data only when it has a registered fine-volume `target_path`.

To seed a manifest from the previous nanoCosmos SSL YAML, export only its
already-4-nm targets:

```bash
nanocosmos-fw-manifest from-legacy-config \
  /path/to/nanoCosmos/configs/nanocosmos-4B-pretrain.yaml \
  manifests/cosem_4nm.jsonl
nanocosmos-fw-manifest validate manifests/cosem_4nm.jsonl
```

## Step 3 — Prepare Cosmos checkpoints

The SFT recipe follows the Cosmos Framework checkpoint flow:

```bash
cd /path/to/cosmos-framework
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Edge \
  -o /home/nvidia/checkpoints/Cosmos3-Edge

uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --local-dir /home/nvidia/checkpoints/wan22_vae
```

## Step 4 — Launch SFT

```bash
cd /path/to/nanoCosmosFW
export EM_MANIFEST_PATH="$PWD/manifests/cosem_4nm.jsonl"
export BASE_CHECKPOINT_PATH=/home/nvidia/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=/home/nvidia/checkpoints/wan22_vae/Wan2.2_VAE.pth
export OUTPUT_ROOT="$HOME/outputs/train"
export NPROC_PER_NODE=8

bash examples/launch_anisotropic_em_transfer_sft_edge.sh
```

The default recipe crops a 256x256x256 volume. After orientation and
thick-section degradation, the last section is repeated once so both videos
are 257 frames, the next Wan length of the form `4n+1`. At 4 nm that crop
covers a 1024 nm cube. Override a setting after the launcher's TOML argument:

```bash
bash examples/launch_anisotropic_em_transfer_sft_edge.sh -- trainer.max_iter=100
```

## Step 5 — Resume or inspect outputs

Cosmos Framework writes DCP checkpoints under:

```text
$OUTPUT_ROOT/nanocosmos/sft/anisotropic_em_transfer_sft_edge/checkpoints/
```

Resume by setting `BASE_CHECKPOINT_PATH` to an `iter_<N>` directory and pass
`checkpoint.load_training_state=true` as a tail override. This restores model,
optimizer, scheduler, trainer, and per-rank RNG state.

## Augmentation contract

The online control synthesis intentionally matches anisotropic EM acquisition:

1. Average fine z slabs into a log-uniform 2x–10x thicker section.
2. Translate each section in-plane by up to two pixels.
3. Randomly zero one section.
4. Apply section-level noise and gamma variation.
5. Return to the fine-grid depth with nearest interpolation.

Both control and target receive identical xy flips, xy transpose, and z reversal.
These are orientation augmentations; no spatial interpolation is used for them.
