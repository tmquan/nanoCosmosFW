"""Create and validate JSONL manifests for paired anisotropic EM SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _volume_path(root: Path, stem: str) -> Path:
    for extension in (".h5", ".hdf5", ".tif", ".tiff", ".npy"):
        candidate = root / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No HDF5, TIFF, or NPY volume named {stem!r} under {root}")


def from_legacy_config(config_path: Path, output_path: Path) -> int:
    """Export image-only SSL entries whose native grid is at least as fine as 4 nm.

    The legacy recipe includes genuinely coarse sources (for example 40x4x4
    CREMI). They are useful controls, but cannot provide a 4 nm supervision
    target by themselves, so they are intentionally excluded here. Add them
    later only with a registered ``target_path`` at the fine resolution.
    """
    import yaml

    config = yaml.safe_load(config_path.read_text())
    volumes = config["data"]["branches"]["ssl"]["volumes"]
    records: list[dict[str, Any]] = []
    for volume in volumes:
        resolution = [float(v) for v in volume["native_resolution"]]
        if any(v > 4 for v in resolution):
            continue
        root = Path(str(volume["root"]).replace("${oc.env:HOME}", str(Path.home())))
        try:
            target = _volume_path(root, str(volume["vol"]))
        except FileNotFoundError:
            continue
        records.append(
            {
                "uuid": str(volume["vol"]),
                "target_path": str(target),
                "target_resolution_nm": resolution,
                "imaging": volume.get("imaging", "electron microscopy"),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return len(records)


def validate(path: Path) -> int:
    """Check manifest structure and referenced files without initializing GPUs."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records in {path}")
    for index, record in enumerate(records):
        for key in ("target_path", "target_resolution_nm"):
            if key not in record:
                raise ValueError(f"Record {index} is missing {key!r}.")
        target = Path(record["target_path"])
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file():
            raise FileNotFoundError(f"Record {index} target is missing: {target}")
        if len(record["target_resolution_nm"]) != 3:
            raise ValueError(f"Record {index} target_resolution_nm must be [z, y, x].")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("from-legacy-config")
    export.add_argument("config", type=Path)
    export.add_argument("output", type=Path)
    check = commands.add_parser("validate")
    check.add_argument("manifest", type=Path)
    args = parser.parse_args()
    count = (
        from_legacy_config(args.config, args.output)
        if args.command == "from-legacy-config"
        else validate(args.manifest)
    )
    print(f"{args.command}: {count} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
