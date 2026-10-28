"""Hydra experiment registration for anisotropic EM Transfer SFT."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from nanocosmos_fw.dataset import get_anisotropic_em_transfer_dataset


def register_experiment() -> None:
    """Register the extension before Cosmos Framework resolves the TOML SKU."""
    store = ConfigStore.instance()
    name = "anisotropic_em_transfer_sft_edge"
    # ConfigStore has no public "contains" API. Re-registering within a process
    # is harmlessly avoided with a package-local sentinel.
    if getattr(register_experiment, "_registered", False):
        return

    model = copy.deepcopy(EDGE_MODEL_CONFIG)
    model.update(
        action_gen=False,
        resolution="256",
        max_num_tokens_after_packing=16384,
        causal_training_strategy="none",
    )
    experiment = LazyDict(
        dict(
            defaults=[
                {"override /model": "mot_fsdp"},
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /optimizer": "adamw"},
                {"override /scheduler": "lambdacosine"},
                {"override /checkpoint": "s3"},
                {"override /callbacks": ["basic", "optimization", "job_monitor", "generation"]},
                {"override /ema": "power"},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": None},
                {"override /vlm_config": None},
                {"override /ckpt_type": "dcp"},
                "_self_",
            ],
            job=dict(project="nanocosmos", group="sft", name=name, wandb_mode="disabled"),
            model=dict(config=model),
            optimizer=dict(
                betas=[0.9, 0.95],
                eps=1e-6,
                fused=True,
                keys_to_select=[
                    "moe_gen",
                    "time_embedder",
                    "vae2llm",
                    "llm2vae",
                    "k_norm_und_for_gen",
                ],
                lr=1e-4,
                lr_multipliers={},
                optimizer_type="AdamW",
                weight_decay=0,
            ),
            scheduler=dict(
                lr_scheduler_type="LambdaCosine",
                cycle_lengths=[10000],
                f_max=[1.0],
                f_min=[0.0],
                f_start=[0.0],
                warm_up_steps=[500],
                verbosity_interval=0,
            ),
            trainer=dict(
                distributed_parallelism="fsdp",
                grad_accum_iter=1,
                logging_iter=10,
                max_iter=10000,
                max_val_iter=None,
                run_validation=False,
                run_validation_on_start=False,
                save_zero_checkpoint=False,
                seed=42,
                timeout_period=999999999,
                validation_iter=100,
                compile_config=dict(recompile_limit=8, use_duck_shape=False),
                cudnn=dict(benchmark=True, deterministic=False),
                ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
                grad_scaler_args=dict(enabled=False),
                callbacks=dict(
                    compile_tokenizer=dict(enabled=False, compile_after_iterations=3, warmup_resolutions=None),
                    grad_clip=dict(clip_norm=0.1, force_finite=True),
                ),
            ),
            checkpoint=dict(
                load_path="???",
                load_training_state=False,
                strict_resume=True,
                keys_to_skip_loading=["net_ema."],
                save_iter=500,
            ),
            dataloader_train=L(PackingDataLoader)(
                dataset_name="video_transfer",
                max_samples_per_batch=1,
                max_sequence_length=16384,
                patch_spatial=2,
                tokenizer_spatial_compression_factor=16,
                tokenizer_temporal_compression_factor=4,
                sound_latent_fps=0,
                audio_sample_rate=48000,
                dataloader=L(RankPartitionedDataLoader)(
                    batch_size=1,
                    in_order=True,
                    num_workers=4,
                    persistent_workers=True,
                    pin_memory=True,
                    prefetch_factor=4,
                    sampler=None,
                    datasets=dict(
                        em_transfer=dict(
                            ratio=1,
                            dataset=L(get_anisotropic_em_transfer_dataset)(
                                manifest_path="${oc.env:EM_MANIFEST_PATH}",
                                frames=256,
                                side=256,
                                fine_resolution_nm=[4, 4, 4],
                                z_factor_range=[2, 10],
                                jitter_px=2.0,
                                missing_probability=0.3,
                                noise_std=0.01,
                                gamma_range=[0.8, 1.25],
                                tokenizer_config="${model.config.vlm_config.tokenizer}",
                            ),
                        )
                    ),
                ),
            ),
            dataloader_val=None,
            upload_reproducible_setup=False,
        ),
        flags={"allow_objects": True},
    )
    store.store(group="experiment", package="_global_", name=name, node=experiment)
    register_experiment._registered = True


__all__ = ["register_experiment"]
