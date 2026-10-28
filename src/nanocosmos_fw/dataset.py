"""Paired anisotropic-EM data loader for Cosmos Framework Transfer SFT.

The framework's Transfer path consumes ``video=[control, target]``.  This
adapter treats EM depth as video time: a registered fine EM crop is the target
and a physically motivated thick-section degradation is the control.  The
target must be genuinely fine data; upsampling an already-coarse source cannot
create a super-resolution training target.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.local_datasets.sft_dataset import SFTDataset
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.utils import log


def _read_shape(path: Path, key: Optional[str]) -> tuple[int, int, int]:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r", locking=False) as handle:
            dataset = handle[key] if key else next(
                value for value in handle.values() if isinstance(value, h5py.Dataset)
            )
            return tuple(int(v) for v in dataset.shape[-3:])
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return tuple(int(v) for v in tifffile.TiffFile(path).series[0].shape[-3:])
    if path.suffix.lower() == ".npy":
        return tuple(int(v) for v in np.load(path, mmap_mode="r").shape[-3:])
    raise ValueError(f"Unsupported EM volume format: {path}")


def _read_crop(path: Path, key: Optional[str], crop: tuple[slice, slice, slice]) -> np.ndarray:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r", locking=False) as handle:
            dataset = handle[key] if key else next(
                value for value in handle.values() if isinstance(value, h5py.Dataset)
            )
            return np.asarray(dataset[crop], dtype=np.float32)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.memmap(path)[crop], dtype=np.float32)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, mmap_mode="r")[crop], dtype=np.float32)
    raise ValueError(f"Unsupported EM volume format: {path}")


def _normalise(image: torch.Tensor) -> torch.Tensor:
    """Robustly map one EM crop to ``[-1, 1]`` without loading its full volume."""
    lo, hi = torch.quantile(image, torch.tensor([0.01, 0.99], device=image.device))
    return ((image - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1) * 2 - 1


def _resize(video: torch.Tensor, size: Sequence[int], *, mode: str) -> torch.Tensor:
    kwargs = {"mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(video[None], size=tuple(size), **kwargs)[0]


def _vae_frame_count(depth: int, temporal_compression_factor: int) -> int:
    """Smallest Wan length ``T >= depth`` with ``(T - 1)`` divisible by the compression factor."""
    remainder = (depth - 1) % temporal_compression_factor
    if remainder == 0:
        return depth
    return depth + (temporal_compression_factor - remainder)


def _pad_depth(video: torch.Tensor, depth: int) -> torch.Tensor:
    """Repeat the last section until ``video`` has ``depth`` frames."""
    current = video.shape[-3]
    if current == depth:
        return video
    if current > depth:
        raise ValueError(f"Cannot pad depth {current} down to {depth}.")
    tail = video[..., -1:, :, :].expand(*video.shape[:-3], depth - current, *video.shape[-2:])
    return torch.cat((video, tail), dim=-3)


def _degrade_sections(
    target: torch.Tensor,
    *,
    z_factor_range: tuple[float, float],
    jitter_px: float,
    missing_probability: float,
    noise_std: float,
    gamma_range: tuple[float, float],
) -> torch.Tensor:
    """Synthesize a thick-section EM control stack from a fine target crop."""
    _, depth, height, width = target.shape
    factor = math.exp(random.uniform(math.log(z_factor_range[0]), math.log(z_factor_range[1])))
    coarse_depth = max(1, int(round(depth / factor)))
    coarse = F.adaptive_avg_pool3d(target[None], (coarse_depth, height, width))[0]

    if jitter_px > 0 and coarse_depth > 1:
        frames = coarse.permute(1, 0, 2, 3)
        shifts = torch.empty(coarse_depth, 2).uniform_(-jitter_px, jitter_px)
        theta = torch.zeros(coarse_depth, 2, 3)
        theta[:, 0, 0] = theta[:, 1, 1] = 1
        theta[:, 0, 2] = 2 * shifts[:, 1] / max(width - 1, 1)
        theta[:, 1, 2] = 2 * shifts[:, 0] / max(height - 1, 1)
        grid = F.affine_grid(theta, frames.shape, align_corners=False)
        coarse = F.grid_sample(
            frames, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        ).permute(1, 0, 2, 3)

    if coarse_depth > 1 and random.random() < missing_probability:
        coarse[:, random.randrange(coarse_depth)] = 0
    if noise_std > 0:
        coarse = coarse + torch.randn_like(coarse) * random.uniform(0, noise_std)
    gamma = random.uniform(*gamma_range)
    coarse = ((coarse + 1) / 2).clamp(0, 1).pow(gamma) * 2 - 1
    return _resize(coarse, (depth, height, width), mode="nearest")


class AnisotropicEMTransferDataset(SFTDataset):
    """Random paired EM crops emitted in Cosmos Transfer SFT format."""

    def __init__(
        self,
        *args: Any,
        fine_resolution_nm: Sequence[float] = (4, 4, 4),
        frames: int = 256,
        side: int = 256,
        z_factor_range: Sequence[float] = (2, 10),
        jitter_px: float = 2.0,
        missing_probability: float = 0.3,
        noise_std: float = 0.01,
        gamma_range: Sequence[float] = (0.8, 1.25),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fine_resolution_nm = tuple(float(v) for v in fine_resolution_nm)
        # ``frames`` is the real EM crop depth. The Wan encoder then needs
        # ``4n+1`` frames, so a 256-deep cube is replicate-padded to 257
        # after augmentation.
        self.frames = int(frames)
        self.vae_frames = _vae_frame_count(self.frames, self.temporal_compression_factor)
        self.num_video_frames = self.vae_frames
        self.side = int(side)
        self.z_factor_range = (float(z_factor_range[0]), float(z_factor_range[1]))
        self.jitter_px = float(jitter_px)
        self.missing_probability = float(missing_probability)
        self.noise_std = float(noise_std)
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))

    def process_one_sample(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        path = Path(metadata["target_path"])
        key = metadata.get("target_key")
        native_resolution = tuple(float(v) for v in metadata["target_resolution_nm"])
        crop_shape = tuple(
            math.ceil(size * fine / native)
            for size, fine, native in zip(
                (self.frames, self.side, self.side), self.fine_resolution_nm, native_resolution
            )
        )
        shape = _read_shape(path, key)
        if any(dim < crop for dim, crop in zip(shape, crop_shape)):
            log.warning(f"Skipping {metadata['uuid']}: {shape=} is smaller than {crop_shape=}")
            return None
        origin = tuple(random.randrange(dim - crop + 1) for dim, crop in zip(shape, crop_shape))
        crop = tuple(slice(start, start + length) for start, length in zip(origin, crop_shape))
        raw = torch.from_numpy(_read_crop(path, key, crop)).unsqueeze(0)
        target = _normalise(_resize(raw, (self.frames, self.side, self.side), mode="trilinear"))

        if random.random() < 0.5:
            target = target.flip(-1)
        if random.random() < 0.5:
            target = target.flip(-2)
        if random.random() < 0.5:
            target = target.transpose(-1, -2)
        if random.random() < 0.5:
            target = target.flip(-3)

        control = _degrade_sections(
            target,
            z_factor_range=self.z_factor_range,
            jitter_px=self.jitter_px,
            missing_probability=self.missing_probability,
            noise_std=self.noise_std,
            gamma_range=self.gamma_range,
        )
        # Pad after orientation and degradation so the extra frame is not
        # averaged into the thick-section control.
        control = _pad_depth(control, self.vae_frames)
        target = _pad_depth(target, self.vae_frames)
        # Cosmos VAE expects RGB video, while EM is a single intensity channel.
        control, target = control.repeat(3, 1, 1, 1), target.repeat(3, 1, 1, 1)
        imaging = metadata.get("imaging", "electron microscopy")
        caption = (
            f"Restore anisotropic {imaging} sections from a thick-section control. "
            f"Target voxel size is z{self.fine_resolution_nm[0]:g} "
            f"y{self.fine_resolution_nm[1]:g} x{self.fine_resolution_nm[2]:g} nm."
        )
        text_ids, caption = self._tokenize_caption(caption)
        image_size = torch.tensor([self.side, self.side, self.side, self.side], dtype=torch.float32)
        return {
            "__key__": metadata["uuid"],
            "__url__": str(path),
            "video": [control, target],
            "dataset_name": "video_transfer",
            "fps": 1.0,
            "conditioning_fps": 1.0,
            "num_frames": self.vae_frames,
            "padding_mask": torch.zeros((1, self.side, self.side), dtype=torch.float32),
            "image_size": [image_size, image_size],
            "ai_caption": caption,
            "text_token_ids": torch.tensor(text_ids),
            "sequence_plan": SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[],
                share_vision_temporal_positions=True,
            ),
        }


def get_anisotropic_em_transfer_dataset(
    manifest_path: str,
    *,
    tokenizer_config: Optional[Any] = None,
    frames: int = 256,
    side: int = 256,
    fine_resolution_nm: Sequence[float] = (4, 4, 4),
    temporal_compression_factor: int = 4,
    **kwargs: Any,
) -> AnisotropicEMTransferDataset:
    """Load a JSONL EM manifest and expose it as a Transfer SFT dataset."""
    manifest = Path(manifest_path)
    metadata = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not metadata:
        raise ValueError(f"No records in EM manifest: {manifest}")
    for index, record in enumerate(metadata):
        record.setdefault("uuid", f"{manifest.stem}-{index}")
        record["target_path"] = str((manifest.parent / record["target_path"]).resolve()) if not Path(
            record["target_path"]
        ).is_absolute() else record["target_path"]
        # The inherited rank-partitioning iterator uses this field only for
        # accounting/logging; the EM reader consumes ``target_path`` above.
        record.setdefault("vision_path", record["target_path"])
        if "target_resolution_nm" not in record:
            raise ValueError(f"Manifest record {record['uuid']} is missing target_resolution_nm.")
    return AnisotropicEMTransferDataset(
        metadata=metadata,
        num_video_frames=frames,
        resolution="256",
        s3_credentials={},
        tokenizer_config=tokenizer_config,
        temporal_compression_factor=temporal_compression_factor,
        fine_resolution_nm=fine_resolution_nm,
        frames=frames,
        side=side,
        **kwargs,
    )


__all__ = ["AnisotropicEMTransferDataset", "get_anisotropic_em_transfer_dataset"]
