#!/usr/bin/env bash
set -euo pipefail

# Required:
#   EM_MANIFEST_PATH       JSONL records describing registered fine EM volumes
#   BASE_CHECKPOINT_PATH   Cosmos3-Edge DCP checkpoint
#   WAN_VAE_PATH           Wan2.2 VAE checkpoint
#
# Optional:
#   NPROC_PER_NODE         GPU processes (default: 8)
#   OUTPUT_ROOT            Framework output root (default: outputs/train)

: "${EM_MANIFEST_PATH:?Set EM_MANIFEST_PATH to an EM JSONL manifest.}"
: "${BASE_CHECKPOINT_PATH:?Set BASE_CHECKPOINT_PATH to a Cosmos3-Edge DCP checkpoint.}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to Wan2.2_VAE.pth.}"
: "${NPROC_PER_NODE:=8}"
: "${OUTPUT_ROOT:=outputs/train}"

test -f "$EM_MANIFEST_PATH" || {
  echo "EM manifest does not exist: $EM_MANIFEST_PATH" >&2
  exit 1
}

IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=src:${PYTHONPATH:-} \
torchrun --nproc_per_node="$NPROC_PER_NODE" -m nanocosmos_fw.train \
  --sft-toml=examples/toml/anisotropic_em_transfer_sft_edge.toml "$@"
