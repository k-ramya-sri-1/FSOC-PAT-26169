"""Read-only aggregation and comparison of completed benchmark reports."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

from benchmark.executor import BenchmarkReport


@dataclass(frozen=True)
class BenchmarkComparison:
    """Preserve and expose a deterministic ordered collection of reports."""

    reports: Tuple[BenchmarkReport, ...]

    def __post_init__(self) -> None:
        reports = tuple(self.reports)
        if not reports:
            raise ValueError("reports must contain at least one BenchmarkReport")
        if any(not isinstance(report, BenchmarkReport) for report in reports):
            raise TypeError("reports must contain only BenchmarkReport objects")
        object.__setattr__(self, "reports", reports)

    @classmethod
    def from_reports(
        cls,
        reports: Sequence[BenchmarkReport],
    ) -> "BenchmarkComparison":
        """Validate reports and retain their order in an immutable tuple."""
        if not isinstance(reports, Sequence):
            raise TypeError("reports must be a sequence of BenchmarkReport objects")
        report_tuple = tuple(reports)
        if not report_tuple:
            raise ValueError("reports must contain at least one BenchmarkReport")
        if any(not isinstance(report, BenchmarkReport) for report in report_tuple):
            raise TypeError("reports must contain only BenchmarkReport objects")
        return cls(report_tuple)

    def scenario_names(self) -> Tuple[str, ...]:
        """Return scenario names in the original report order."""
        return tuple(report.scenario_name for report in self.reports)

    def metric_table(self) -> Tuple[Mapping[str, object], ...]:
        """Return one read-only row of existing report data per scenario."""
        return tuple(
            MappingProxyType(
                {
                    "scenario_name": report.scenario_name,
                    "source_type": report.source_type,
                    "frame_count": report.frame_count,
                    "elapsed_simulation_seconds": report.elapsed_simulation_seconds,
                    "motion_type": report.motion_type,
                    "seed": report.seed,
                    "metrics": MappingProxyType(dict(report.metrics)),
                }
            )
            for report in self.reports
        )

    def deterministic_signature(self) -> tuple[object, ...]:
        """Return ordered report signatures without introducing new timing data."""
        return tuple(report.deterministic_signature() for report in self.reports)
