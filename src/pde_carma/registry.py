"""Immutable registry of original implementation entry points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    directory: str
    train_entry: str
    infer_entry: str
    carma_supported: bool
    reproduction_scripts: tuple[str, ...]


VARIANTS: dict[str, VariantSpec] = {
    "dit-unidirectional": VariantSpec(
        key="dit-unidirectional",
        label="Unidirectional DiT",
        directory="dit_unidirectional",
        train_entry="train_dit_unidirectional.py",
        infer_entry="infer_dit.py",
        carma_supported=False,
        reproduction_scripts=(
            "scripts/evaluate_unidirectional_dit_multirun.py",
        ),
    ),
    "dit-bidirectional": VariantSpec(
        key="dit-bidirectional",
        label="Bidirectional DiT",
        directory="dit_bidirectional",
        train_entry="train_dit.py",
        infer_entry="infer_dit.py",
        carma_supported=True,
        reproduction_scripts=(
            "scripts/generate_overleaf_table_results_dit.py",
            "scripts/generate_tensors_plot_error_bar_dit.py",
            "scripts/generate_tensors_plot_error_bar_dit_multirun.py",
            "scripts/plot_carma_rollouts.py",
            "scripts/replot_dataset_cyclic_distribution_dit.py",
        ),
    ),
    "flowformer-unidirectional": VariantSpec(
        key="flowformer-unidirectional",
        label="Unidirectional FlowFormer",
        directory="flowformer_unidirectional",
        train_entry="train_flowformer.py",
        infer_entry="infer_flowformer.py",
        carma_supported=False,
        reproduction_scripts=(
            "scripts/generate_unifwd_table3.py",
        ),
    ),
    "flowformer-bidirectional": VariantSpec(
        key="flowformer-bidirectional",
        label="Bidirectional FlowFormer",
        directory="flowformer_bidirectional",
        train_entry="train_flowformer.py",
        infer_entry="infer_flowformer.py",
        carma_supported=True,
        reproduction_scripts=(
            "run_ode_inference_sweep.py",
            "run_ode_inference_sweep_resume.py",
            "scripts/generate_adaptive_beam_cyclic_results.py",
            "scripts/generate_adaptive_variance_cyclic_results.py",
            "scripts/generate_corrected_adaptive_cyclic_results.py",
            "scripts/generate_greedy_oracle_table_results.py",
            "scripts/generate_overleaf_table_results.py",
            "scripts/generate_strong_greedy_cyclic_results.py",
            "scripts/generate_tensors_plot_error_bar.py",
            "scripts/generate_tensors_plot_error_bar_multirun.py",
            "scripts/plot_carma_rollouts.py",
            "scripts/replot_dataset_cyclic_distribution.py",
            "scripts/run_ode_inference_sweep.py",
            "scripts/run_ode_inference_sweep_resume.py",
        ),
    ),
}


def get_variant(key: str) -> VariantSpec:
    try:
        return VARIANTS[key]
    except KeyError as exc:
        choices = ", ".join(VARIANTS)
        raise ValueError(f"Unknown variant {key!r}; choose one of: {choices}") from exc

