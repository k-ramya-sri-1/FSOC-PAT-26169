from dataclasses import replace

import pytest

from benchmark.comparison import BenchmarkComparison
from benchmark.executor import BenchmarkReport
from metrics.evaluator import ProcessingStatistics


def _reports():
    return (
        BenchmarkReport(
            scenario_name="straight",
            source_type="simulation",
            frame_count=12,
            elapsed_simulation_seconds=0.4,
            metrics={
                "processing": ProcessingStatistics(12, 0.01, 100.0),
                "lock_retention_percentage": 75.0,
            },
            motion_type="straight",
            seed=19,
        ),
        BenchmarkReport(
            scenario_name="circular",
            source_type="simulation",
            frame_count=12,
            elapsed_simulation_seconds=0.4,
            metrics={
                "processing": ProcessingStatistics(12, 0.02, 50.0),
                "lock_retention_percentage": 50.0,
            },
            motion_type="circular",
            seed=19,
        ),
    )


def test_from_reports_accepts_reports_and_preserves_order():
    reports = _reports()

    comparison = BenchmarkComparison.from_reports(list(reports))

    assert comparison.reports == reports
    assert isinstance(comparison.reports, tuple)
    assert comparison.scenario_names() == ("straight", "circular")


def test_from_reports_rejects_empty_or_invalid_entries():
    with pytest.raises(ValueError):
        BenchmarkComparison.from_reports(())
    with pytest.raises(TypeError):
        BenchmarkComparison.from_reports(("not a report",))
    with pytest.raises(TypeError):
        BenchmarkComparison.from_reports(None)


def test_metric_table_has_ordered_read_only_rows_with_report_fields():
    reports = _reports()
    comparison = BenchmarkComparison.from_reports(reports)

    rows = comparison.metric_table()

    assert isinstance(rows, tuple)
    assert len(rows) == len(reports)
    assert [row["scenario_name"] for row in rows] == ["straight", "circular"]
    for row, report in zip(rows, reports):
        assert row["scenario_name"] == report.scenario_name
        assert row["source_type"] == report.source_type
        assert row["frame_count"] == report.frame_count
        assert row["elapsed_simulation_seconds"] == report.elapsed_simulation_seconds
        assert row["motion_type"] == report.motion_type
        assert row["seed"] == report.seed
        assert dict(row["metrics"]) == dict(report.metrics)
        with pytest.raises(TypeError):
            row["frame_count"] = 0
        with pytest.raises(TypeError):
            row["metrics"]["lock_retention_percentage"] = 0.0


def test_comparison_does_not_mutate_reports_or_their_metrics():
    reports = _reports()
    before = tuple((report.deterministic_signature(), dict(report.metrics)) for report in reports)

    comparison = BenchmarkComparison.from_reports(reports)
    comparison.scenario_names()
    comparison.metric_table()
    comparison.deterministic_signature()

    after = tuple((report.deterministic_signature(), dict(report.metrics)) for report in reports)
    assert after == before


def test_equivalent_deterministic_reports_have_identical_signatures():
    reports = _reports()
    timing_variant = replace(
        reports[0],
        metrics={
            **reports[0].metrics,
            "processing": ProcessingStatistics(12, 5.0, 0.2),
        },
    )
    first = BenchmarkComparison.from_reports(reports)
    second = BenchmarkComparison.from_reports((timing_variant, reports[1]))

    assert first.deterministic_signature() == second.deterministic_signature()


def test_comparison_preserves_scenario_order_without_ranking():
    reports = _reports()
    reversed_comparison = BenchmarkComparison.from_reports(tuple(reversed(reports)))

    assert reversed_comparison.scenario_names() == ("circular", "straight")
    assert reversed_comparison.deterministic_signature() == tuple(
        report.deterministic_signature() for report in reversed(reports)
    )
