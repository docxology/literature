"""Quality metrics tracking for summarization system.

Tracks and analyzes quality metrics across all summarization operations
to identify trends, failure patterns, and system health.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

from infrastructure.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a single summarization operation."""

    citation_key: str
    timestamp: float
    success: bool
    quality_score: float
    validation_errors: List[str]
    validation_warnings: List[str]
    input_chars: int
    input_words: int
    output_words: int
    generation_time: float
    attempts: int
    paper_classification: Optional[str] = None

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.output_words / max(1, self.input_words)

    @property
    def words_per_second(self) -> float:
        """Calculate generation speed."""
        return self.output_words / max(0.001, self.generation_time)

    @property
    def has_hard_failure(self) -> bool:
        """Check if this operation had a hard validation failure."""
        hard_failure_keywords = [
            "title mismatch",
            "major hallucination",
            "severe repetition"
        ]
        error_text = " ".join(self.validation_errors).lower()
        return any(keyword in error_text for keyword in hard_failure_keywords)


@dataclass
class AggregatedMetrics:
    """Aggregated quality metrics across multiple operations."""

    total_operations: int = 0
    successful_operations: int = 0
    average_quality_score: float = 0.0
    average_compression_ratio: float = 0.0
    average_generation_speed: float = 0.0
    hard_failure_rate: float = 0.0
    retry_rate: float = 0.0

    # Error pattern analysis
    common_validation_errors: Counter[str] = field(default_factory=lambda: Counter())
    common_validation_warnings: Counter[str] = field(default_factory=lambda: Counter())

    # Performance metrics
    total_input_chars: int = 0
    total_output_words: int = 0
    total_generation_time: float = 0.0

    # Classification distribution
    classification_counts: Counter[str] = field(default_factory=lambda: Counter())

    # Time-based metrics
    operations_by_hour: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    quality_trends: List[float] = field(default_factory=list)

    def update_with_operation(self, metrics: QualityMetrics):
        """Update aggregated metrics with a new operation."""
        self.total_operations += 1
        if metrics.success:
            self.successful_operations += 1

        # Update averages incrementally
        self.average_quality_score = (
            (self.average_quality_score * (self.total_operations - 1) + metrics.quality_score)
            / self.total_operations
        )

        self.average_compression_ratio = (
            (self.average_compression_ratio * (self.total_operations - 1) + metrics.compression_ratio)
            / self.total_operations
        )

        self.average_generation_speed = (
            (self.average_generation_speed * (self.total_operations - 1) + metrics.words_per_second)
            / self.total_operations
        )

        # Update error patterns
        for error in metrics.validation_errors:
            self.common_validation_errors[error] += 1

        for warning in metrics.validation_warnings:
            self.common_validation_warnings[warning] += 1

        # Update performance totals
        self.total_input_chars += metrics.input_chars
        self.total_output_words += metrics.output_words
        self.total_generation_time += metrics.generation_time

        # Update classification counts
        if metrics.paper_classification:
            self.classification_counts[metrics.paper_classification] += 1

        # Update time-based metrics
        hour_key = time.strftime("%Y-%m-%d-%H", time.localtime(metrics.timestamp))
        self.operations_by_hour[hour_key] += 1

        # Keep quality trend (last 100 operations)
        self.quality_trends.append(metrics.quality_score)
        if len(self.quality_trends) > 100:
            self.quality_trends.pop(0)

        # Calculate rates
        self.hard_failure_rate = sum(1 for _ in [metrics] if metrics.has_hard_failure) / self.total_operations
        self.retry_rate = (sum(metrics.attempts - 1 for _ in [metrics]) / self.total_operations) if self.total_operations > 0 else 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_operations / max(1, self.total_operations)

    @property
    def average_input_chars(self) -> float:
        """Calculate average input characters."""
        return self.total_input_chars / max(1, self.total_operations)

    @property
    def average_output_words(self) -> float:
        """Calculate average output words."""
        return self.total_output_words / max(1, self.total_operations)

    def get_quality_trend(self, window_size: int = 10) -> List[float]:
        """Get recent quality trend with moving average."""
        if len(self.quality_trends) < window_size:
            return self.quality_trends

        trend = []
        for i in range(len(self.quality_trends) - window_size + 1):
            window = self.quality_trends[i:i + window_size]
            trend.append(sum(window) / len(window))
        return trend

    def get_failure_analysis(self) -> Dict[str, Any]:
        """Get detailed failure analysis."""
        return {
            "hard_failure_rate": self.hard_failure_rate,
            "retry_rate": self.retry_rate,
            "most_common_errors": dict(self.common_validation_errors.most_common(5)),
            "most_common_warnings": dict(self.common_validation_warnings.most_common(5)),
            "classification_distribution": dict(self.classification_counts),
            "recent_quality_trend": self.get_quality_trend(20)
        }


class QualityTracker:
    """
    Tracks quality metrics across all summarization operations.

    Provides persistent storage, trend analysis, and health monitoring
    for the summarization system.
    """

    def __init__(self, metrics_file: Optional[Path] = None):
        """Initialize quality tracker.

        Args:
            metrics_file: Path to store metrics (default: data/summarization_quality_metrics.json)
        """
        self.metrics_file = metrics_file or Path("data/summarization_quality_metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory metrics
        self.current_metrics = AggregatedMetrics()
        self.operation_history: List[QualityMetrics] = []

        # Load existing metrics
        self._load_metrics()

    def record_operation(self, metrics: QualityMetrics):
        """Record a summarization operation's quality metrics."""
        self.operation_history.append(metrics)
        self.current_metrics.update_with_operation(metrics)

        # Keep only last 1000 operations in memory
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]

        # Save to disk periodically (every 10 operations)
        if len(self.operation_history) % 10 == 0:
            self._save_metrics()

        logger.info(
            f"Quality tracking: Recorded operation {metrics.citation_key} "
            f"(score: {metrics.quality_score:.2f}, success: {metrics.success})"
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        recent_ops = self.operation_history[-50:] if len(self.operation_history) >= 50 else self.operation_history

        if not recent_ops:
            return {"status": "unknown", "message": "No operations recorded yet"}

        recent_success_rate = sum(1 for op in recent_ops if op.success) / len(recent_ops)
        recent_avg_score = sum(op.quality_score for op in recent_ops) / len(recent_ops)
        recent_hard_failures = sum(1 for op in recent_ops if op.has_hard_failure)

        # Determine health status
        if recent_success_rate >= 0.95 and recent_avg_score >= 0.8 and recent_hard_failures == 0:
            status = "excellent"
        elif recent_success_rate >= 0.9 and recent_avg_score >= 0.7 and recent_hard_failures <= 1:
            status = "good"
        elif recent_success_rate >= 0.8 and recent_avg_score >= 0.6:
            status = "fair"
        else:
            status = "poor"

        return {
            "status": status,
            "recent_success_rate": recent_success_rate,
            "recent_avg_score": recent_avg_score,
            "recent_hard_failures": recent_hard_failures,
            "total_operations": len(self.operation_history),
            "aggregated_metrics": {
                "success_rate": self.current_metrics.success_rate,
                "average_quality_score": self.current_metrics.average_quality_score,
                "hard_failure_rate": self.current_metrics.hard_failure_rate,
                "retry_rate": self.current_metrics.retry_rate
            }
        }

    def get_failure_analysis(self) -> Dict[str, Any]:
        """Get comprehensive failure analysis."""
        return self.current_metrics.get_failure_analysis()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        return {
            "total_operations": self.current_metrics.total_operations,
            "average_input_chars": self.current_metrics.average_input_chars,
            "average_output_words": self.current_metrics.average_output_words,
            "average_compression_ratio": self.current_metrics.average_compression_ratio,
            "average_generation_speed": self.current_metrics.average_generation_speed,
            "total_processing_time": self.current_metrics.total_generation_time
        }

    def export_metrics(self, output_file: Path):
        """Export all metrics to JSON file."""
        export_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_operations": len(self.operation_history),
                "metrics_file": str(self.metrics_file)
            },
            "aggregated_metrics": {
                "total_operations": self.current_metrics.total_operations,
                "successful_operations": self.current_metrics.successful_operations,
                "success_rate": self.current_metrics.success_rate,
                "average_quality_score": self.current_metrics.average_quality_score,
                "average_compression_ratio": self.current_metrics.average_compression_ratio,
                "average_generation_speed": self.current_metrics.average_generation_speed,
                "hard_failure_rate": self.current_metrics.hard_failure_rate,
                "retry_rate": self.current_metrics.retry_rate,
                "total_input_chars": self.current_metrics.total_input_chars,
                "total_output_words": self.current_metrics.total_output_words,
                "total_generation_time": self.current_metrics.total_generation_time
            },
            "error_patterns": {
                "common_validation_errors": dict(self.current_metrics.common_validation_errors.most_common(10)),
                "common_validation_warnings": dict(self.current_metrics.common_validation_warnings.most_common(10))
            },
            "classification_distribution": dict(self.current_metrics.classification_counts),
            "recent_operations": [
                {
                    "citation_key": op.citation_key,
                    "timestamp": op.timestamp,
                    "success": op.success,
                    "quality_score": op.quality_score,
                    "has_hard_failure": op.has_hard_failure,
                    "attempts": op.attempts,
                    "paper_classification": op.paper_classification
                }
                for op in self.operation_history[-100:]  # Last 100 operations
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported quality metrics to {output_file}")

    def _load_metrics(self):
        """Load metrics from disk."""
        if not self.metrics_file.exists():
            return

        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)

            # Reconstruct aggregated metrics
            agg_data = data.get("aggregated_metrics", {})
            self.current_metrics = AggregatedMetrics(
                total_operations=agg_data.get("total_operations", 0),
                successful_operations=agg_data.get("successful_operations", 0),
                average_quality_score=agg_data.get("average_quality_score", 0.0),
                average_compression_ratio=agg_data.get("average_compression_ratio", 0.0),
                average_generation_speed=agg_data.get("average_generation_speed", 0.0),
                hard_failure_rate=agg_data.get("hard_failure_rate", 0.0),
                retry_rate=agg_data.get("retry_rate", 0.0),
                total_input_chars=agg_data.get("total_input_chars", 0),
                total_output_words=agg_data.get("total_output_words", 0),
                total_generation_time=agg_data.get("total_generation_time", 0.0)
            )

            # Load error patterns
            error_patterns = data.get("error_patterns", {})
            self.current_metrics.common_validation_errors = Counter(error_patterns.get("common_validation_errors", {}))
            self.current_metrics.common_validation_warnings = Counter(error_patterns.get("common_validation_warnings", {}))

            # Load classification distribution
            self.current_metrics.classification_counts = Counter(data.get("classification_distribution", {}))

            # Load recent operation history
            recent_ops = data.get("recent_operations", [])
            for op_data in recent_ops:
                op = QualityMetrics(
                    citation_key=op_data["citation_key"],
                    timestamp=op_data["timestamp"],
                    success=op_data["success"],
                    quality_score=op_data["quality_score"],
                    validation_errors=[],  # Not stored in summary
                    validation_warnings=[],  # Not stored in summary
                    input_chars=0,  # Not stored in summary
                    input_words=0,  # Not stored in summary
                    output_words=0,  # Not stored in summary
                    generation_time=0.0,  # Not stored in summary
                    attempts=op_data["attempts"],
                    paper_classification=op_data.get("paper_classification")
                )
                self.operation_history.append(op)

            logger.info(f"Loaded quality metrics from {self.metrics_file}")

        except Exception as e:
            logger.warning(f"Failed to load quality metrics: {e}")
            # Reset to empty state
            self.current_metrics = AggregatedMetrics()
            self.operation_history = []

    def _save_metrics(self):
        """Save current metrics to disk."""
        try:
            export_data = {
                "metadata": {
                    "last_updated": time.time(),
                    "total_operations": len(self.operation_history)
                },
                "aggregated_metrics": {
                    "total_operations": self.current_metrics.total_operations,
                    "successful_operations": self.current_metrics.successful_operations,
                    "success_rate": self.current_metrics.success_rate,
                    "average_quality_score": self.current_metrics.average_quality_score,
                    "average_compression_ratio": self.current_metrics.average_compression_ratio,
                    "average_generation_speed": self.current_metrics.average_generation_speed,
                    "hard_failure_rate": self.current_metrics.hard_failure_rate,
                    "retry_rate": self.current_metrics.retry_rate,
                    "total_input_chars": self.current_metrics.total_input_chars,
                    "total_output_words": self.current_metrics.total_output_words,
                    "total_generation_time": self.current_metrics.total_generation_time
                },
                "error_patterns": {
                    "common_validation_errors": dict(self.current_metrics.common_validation_errors.most_common(10)),
                    "common_validation_warnings": dict(self.current_metrics.common_validation_warnings.most_common(10))
                },
                "classification_distribution": dict(self.current_metrics.classification_counts),
                "recent_operations": [
                    {
                        "citation_key": op.citation_key,
                        "timestamp": op.timestamp,
                        "success": op.success,
                        "quality_score": op.quality_score,
                        "has_hard_failure": op.has_hard_failure,
                        "attempts": op.attempts,
                        "paper_classification": op.paper_classification
                    }
                    for op in self.operation_history[-100:]  # Last 100 operations
                ]
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(export_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save quality metrics: {e}")
