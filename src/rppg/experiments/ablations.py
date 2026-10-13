"""Independent controlled model ablations; available manuscript rows are not fabricated."""
from dataclasses import replace

ABLATIONS = {
    "full": {},
    "no_ssl": {"use_ssl": False},
    "no_rope": {"use_rope": False},
    "no_long_range": {"use_long_range": False},
    "no_memory": {"use_memory": False},
    "no_attention": {"use_memory": False, "use_long_range": False},
    "no_fft": {"use_fft": False},
    "fft_weight_0_5": {"fft_weight": 0.5},
    "fft_weight_3_0": {"fft_weight": 3.0},
}


def ablation_config(config, variant):
    if variant not in ABLATIONS:
        raise ValueError(f"Unknown ablation {variant}; available {sorted(ABLATIONS)}")
    return replace(config, **ABLATIONS[variant])
