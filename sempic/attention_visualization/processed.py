"""Render compact model-level reports from validated processed records."""

from __future__ import annotations

from pathlib import Path

from sempic.attention_metrics.processed_storage import load_processed_metrics

from .attention_reports import (
    plot_attention_reports,
    validate_attention_report_records,
)
from .retrieval_reports import (
    plot_retrieval_reports,
    validate_retrieval_report_records,
)


def plot_processed_metrics(
    metrics_path: str | Path, output_dir: str | Path
) -> list[Path]:
    """Render five compact multi-page reports for every represented model."""
    artifact = load_processed_metrics(metrics_path)
    outputs = []
    root = Path(output_dir)
    model_ids = sorted({record["model_id"] for record in artifact["records"]})
    records_by_model = {}
    for model_id in model_ids:
        records = [
            record for record in artifact["records"]
            if record["model_id"] == model_id
        ]
        validate_attention_report_records(records, artifact["processing_config"])
        validate_retrieval_report_records(records)
        records_by_model[model_id] = records

    for model_id, records in records_by_model.items():
        outputs.extend(
            plot_attention_reports(
                records, artifact["metric_specs"], root / model_id,
                model_id=model_id,
                processing_config=artifact["processing_config"],
                fingerprint=artifact["processing_fingerprint"],
            )
        )
        outputs.extend(
            plot_retrieval_reports(
                records, artifact["metric_specs"], root / model_id,
                model_id=model_id,
                fingerprint=artifact["processing_fingerprint"],
            )
        )
    return outputs


__all__ = ["plot_processed_metrics"]
