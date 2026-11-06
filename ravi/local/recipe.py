"""Versioned effective local training recipe; changes invalidate continuation."""
from .catalog import PROFILES, crop_latents

RECIPE_VERSION = 'local-mlx-v5-cloud-matched'

# v5 matches the cloud recipe whose adapters the composer liked (configs/sa3/B_tier1.yaml,
# runs_local/adapters/B_tier1 metadata): every DiT layer except the duration conditioner.
# v4 additionally protected nine modules; the liked cloud adapter adapted all of them.
EXCLUDED_TARGETS = ('seconds_total',)

# Trigger token prepended to every training caption and, for adapters trained with it,
# to generation prompts. Same token as the cloud arms.
TRIGGER = 'rvkr'

TARGET_LUFS = -16.0
TRUE_PEAK_CEILING_DBTP = -1.0
MIN_EXAMPLE_SECONDS = 1.0


def training_recipe(profile):
    p = PROFILES[profile]
    return {
        'name': RECIPE_VERSION,
        # Full generation uses mask channel 0, as torch inference/training and MLX generation do.
        'gen_mask': 'zeros',
        'mask': 'full only; section-edit retention unvalidated',
        'duration_conditioner': 'frozen',
        'normalization': f'one gain per track to {TARGET_LUFS:g} LUFS, true peak <= {TRUE_PEAK_CEILING_DBTP:g} dBTP',
        'examples': f"whole described regions, split only above {p['example_max_seconds']} s; no clip tiling",
        'caption_augmentation': 'none',
        'trigger': TRIGGER,
        'crop_seconds': p['crop_seconds'], 'crop_latents': crop_latents(profile),
        'random_crop': True, 'padding': 'encoded silence latent',
        'rank': p['rank'], 'alpha': p['rank'], 'adapter_type': 'dora-rows',
        'exclude': list(EXCLUDED_TARGETS),
        'learning_rate': 1e-4, 'batch_size': 1,
        'beta1': .9, 'beta2': .95, 'eps': 1e-8, 'weight_decay': .01,
        'lr_scheduler': 'none', 'gradient_clip_val': 0.,
        'timestep_sampler': 'trunc_logit_normal', 'dist_shift': 'full',
        'use_effective_length': True, 'cfg_dropout_prob': .1,
        'optimizer_resume': False,
        # Deliberate, recorded differences from the cloud B_tier1 run.
        'differences_from_cloud': [
            'MLX trains fp16 base weights; the cloud used bf16 (MLX trainer has no bf16 path)',
            'full-generation mask only; the cloud also sampled 10% random and 10% causal inpaint masks',
            'no caption augmentation: local captions are single prose fields, so field shuffle/drop do not apply',
            'window and rank follow the workload profile; only the 32gb profile matches the cloud 47 s / rank 16',
        ],
    }
