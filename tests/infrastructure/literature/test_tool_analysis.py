"""Tests for infrastructure.literature.tool_analysis (moved from scripts/)."""

import json

import pytest

from infrastructure.literature.tool_analysis import (
    cooccurrence,
    common,
    language_trends,
    over_time,
    repo_extractor,
    run_comparison,
    specific_tools,
)

SAMPLE_TEXT = (
    "Deep reinforcement learning for active inference\n"
    "We implement the model in Python.\n"
    "Code: https://github.com/example/deep-rl, 2024.\n"
    "The stack uses PyTorch and NumPy.\n"
    "Also Jupyter notebooks.\n"
)


@pytest.fixture
def extracted_dir(tmp_path):
    d = tmp_path / "extracted_text"
    d.mkdir()
    (d / "smith2024deep.txt").write_text(SAMPLE_TEXT, encoding="utf-8")
    return d


class TestCleanUrl:
    def test_strips_trailing_punctuation(self):
        assert common.clean_url("https://github.com/user/repo.") == "https://github.com/user/repo"
        assert common.clean_url("https://github.com/user/repo,") == "https://github.com/user/repo"

    def test_strips_trailing_year_and_version(self):
        assert common.clean_url("https://github.com/user/repo,2024") == "https://github.com/user/repo"
        assert common.clean_url("https://github.com/user/repo,2.0") == "https://github.com/user/repo"

    def test_strips_query_fragment_after_semicolon(self):
        assert common.clean_url("https://github.com/user/repo;visit=x") == "https://github.com/user/repo"

    def test_adds_scheme_to_bare_url(self):
        assert common.clean_url("github.com/user/repo") == "https://github.com/user/repo"

    def test_rejects_incomplete_urls(self):
        assert common.clean_url("https://github.com/a") is None
        assert common.clean_url("https://github.com/user/repo[") is None
        # Trailing dash is stripped before the endswith('-') guard (original behavior).
        assert common.clean_url("https://github.com/user/repo-") == "https://github.com/user/repo"


class TestExtractYear:
    def test_valid_year(self):
        assert common.extract_year_from_citation_key("smith2024deep") == 2024

    def test_out_of_range_year(self):
        assert common.extract_year_from_citation_key("paper1800x") is None
        assert common.extract_year_from_citation_key("paper2200x") is None

    def test_no_year(self):
        assert common.extract_year_from_citation_key("noyear") is None


class TestFindGithubUrls:
    def test_finds_and_cleans_urls(self):
        urls = common.find_github_urls("See https://github.com/user/repo, and github.com/other/pkg.")
        assert urls == ["https://github.com/user/repo", "https://github.com/other/pkg"]

    def test_deduplicates(self):
        content = "https://github.com/user/repo. Again: https://github.com/user/repo,"
        assert common.find_github_urls(content) == ["https://github.com/user/repo"]

    def test_no_urls(self):
        assert common.find_github_urls("no repos here") == []


class TestMentionSearch:
    def test_code_mentions(self):
        lines = SAMPLE_TEXT.split("\n")
        urls = common.find_github_urls(SAMPLE_TEXT)
        mentions = common.find_code_mentions(SAMPLE_TEXT, urls, lines)
        # code_patterns covers languages + ML frameworks; NumPy/Jupyter belong to
        # the specific-tools table, not the co-occurrence language table.
        assert {"Python", "PyTorch"} == {m for m in mentions if m in ("Python", "PyTorch")}
        assert "NumPy" not in mentions and "Jupyter" not in mentions

    def test_context_window_finds_mentions_near_url(self):
        lines = ["https://github.com/user/repo", "this project uses PyTorch"]
        found = common.find_items_in_content(
            "https://github.com/user/repo this project uses PyTorch",
            specific_tools.specific_tools, ["https://github.com/user/repo"], lines, context_lines=10,
        )
        assert "PyTorch" in found

    def test_custom_patterns(self):
        lines = ["uses ROS2 here"]
        found = common.find_items_in_content("uses ROS2 here", common.code_patterns, [], lines)
        assert "ROS2" in found


class TestExtractData:
    def test_specific_tools(self, extracted_dir):
        data = specific_tools.extract_data(extracted_dir)
        assert len(data) == 1
        entry = data[0]
        assert entry["paper"] == "smith2024deep"
        assert {"PyTorch", "NumPy", "Jupyter"} <= set(entry["tools"])
        assert entry["repo_count"] == 1

    def test_cooccurrence_languages(self, extracted_dir):
        data = cooccurrence.extract_data(extracted_dir)
        assert len(data) == 1
        assert "Python" in data[0]["languages"]

    def test_over_time_requires_year(self, extracted_dir):
        data = over_time.extract_data(extracted_dir)
        assert len(data) == 1
        assert data[0]["year"] == 2024
        assert data[0]["languages"] == {"Python"}

    def test_language_trends_requires_year(self, extracted_dir):
        data = language_trends.extract_data(extracted_dir)
        assert len(data) == 1
        assert data[0]["year"] == 2024

    def test_missing_directory(self, tmp_path):
        assert specific_tools.extract_data(tmp_path / "nope") == []
        assert cooccurrence.extract_data(tmp_path / "nope") == []
        assert over_time.extract_data(tmp_path / "nope") == []

    def test_file_without_repos_skipped(self, tmp_path):
        d = tmp_path / "extracted_text"
        d.mkdir()
        (d / "nopaper.txt").write_text("just prose, no repos", encoding="utf-8")
        assert specific_tools.extract_data(d) == []
        assert repo_extractor.extract_github_info(d) == []


class TestCooccurrence:
    def test_counts(self):
        data = [
            {"languages": {"Python", "R"}, "paper": "a"},
            {"languages": {"Python", "R", "JAX"}, "paper": "b"},
        ]
        tool_counts, pair_counts, triplet_counts = cooccurrence.analyze_cooccurrence(data)
        assert tool_counts["Python"] == 2
        assert tool_counts["JAX"] == 1
        assert pair_counts[("Python", "R")] == 2
        assert triplet_counts[("JAX", "Python", "R")] == 1

    def test_jaccard(self):
        sims = cooccurrence.calculate_jaccard_similarity(
            {("Python", "R"): 2}, {"Python": 2, "R": 2}
        )
        assert sims[("Python", "R")]["jaccard"] == 1.0
        sims = cooccurrence.calculate_jaccard_similarity(
            {("Python", "R"): 1}, {"Python": 2, "R": 2}
        )
        assert sims[("Python", "R")]["jaccard"] == pytest.approx(1 / 3)


class TestTimeSeries:
    def test_prepare_time_series_data(self):
        data = [
            {"year": 2023, "languages": {"Python"}, "tools": {"PyTorch"}, "paper": "a"},
            {"year": 2024, "languages": {"Python", "R"}, "tools": set(), "paper": "b"},
        ]
        years, lang_series, tool_series, lang_totals, tool_totals = over_time.prepare_time_series_data(data)
        assert years == [2023, 2024]
        assert lang_series["Python"] == [1, 1]
        assert lang_series["R"] == [0, 1]
        assert tool_totals["PyTorch"] == 1

    def test_prepare_chart_data(self):
        data = [
            {"year": 2023, "languages": {"Python"}, "paper": "a"},
            {"year": 2024, "languages": {"Python", "R"}, "paper": "b"},
        ]
        years, lang_data, sorted_langs, year_lang_counts = language_trends.prepare_chart_data(data)
        assert years == [2023, 2024]
        assert sorted_langs[0] == "Python"
        assert lang_data["R"] == [0, 1]


class TestMarkdownReport:
    def test_report_contains_paper_and_repo(self):
        results = [{
            "paper": "smith2024deep", "title": "Deep RL", "authors": ["J. Smith"],
            "year": 2024, "venue": "arXiv", "doi": "10.1/x",
            "urls": ["https://github.com/example/deep-rl"],
            "languages": ["Python", "PyTorch"],
        }]
        report = repo_extractor.generate_markdown_report(results)
        assert "### smith2024deep" in report
        assert "- https://github.com/example/deep-rl" in report
        assert "**Paper:** Deep RL" in report

    def test_report_groups_unspecified(self):
        results = [{
            "paper": "nopaper", "title": "", "authors": [], "year": None,
            "venue": None, "doi": None, "urls": ["https://github.com/u/r"],
            "languages": ["Not specified"],
        }]
        report = repo_extractor.generate_markdown_report(results)
        assert "## Not Specified" in report


class TestReadPreviousSummary:
    def test_parses_stats(self, tmp_path):
        plots = tmp_path / "plots"
        plots.mkdir()
        (plots / "tools_and_languages_summary.md").write_text(
            "**Analysis Date:** 2025-01-01\n\n"
            "**Total Papers Analyzed:** 42\n"
            "**Year Range:** 2023 - 2024\n"
            "**Total Unique Languages:** 4\n"
            "**Total Unique Tools:** 9\n",
            encoding="utf-8",
        )
        stats = run_comparison.read_previous_summary(plots)
        assert stats["papers"] == 42
        assert stats["year_min"] == 2023
        assert stats["year_max"] == 2024
        assert stats["languages"] == 4
        assert stats["tools"] == 9

    def test_missing_summary(self, tmp_path):
        assert run_comparison.read_previous_summary(tmp_path) is None


class TestAnalyzeCurrentState:
    def test_counts(self, extracted_dir):
        state = run_comparison.analyze_current_state(extracted_dir)
        assert state["total_files"] == 1
        assert state["files_with_github"] == 1
        assert state["files_with_year"] == 1
        assert state["years"] == {2024: 1}


class TestRunPipelines:
    """End-to-end smoke of each run() entry point against a sandbox."""

    def test_specific_tools_run(self, extracted_dir, tmp_path):
        out = tmp_path / "plots"
        specific_tools.run(extracted_text_dir=extracted_dir, output_dir=out)
        for name in ("specific_tools_analysis.txt", "specific_tools_by_paper.csv",
                     "specific_tools_frequency.csv", "specific_tools.html"):
            assert (out / name).exists()
        assert "PyTorch" in (out / "specific_tools_analysis.txt").read_text()

    def test_cooccurrence_run(self, extracted_dir, tmp_path):
        out = tmp_path / "plots"
        cooccurrence.run(extracted_text_dir=extracted_dir, output_dir=out)
        for name in ("tool_cooccurrence_analysis.txt", "tool_cooccurrence_matrix.csv",
                     "tool_combinations.csv", "tool_cooccurrence.html"):
            assert (out / name).exists()

    def test_over_time_run(self, extracted_dir, tmp_path):
        out = tmp_path / "plots"
        over_time.run(extracted_text_dir=extracted_dir, output_dir=out)
        for name in ("tools_and_languages_summary.md", "languages_over_time.csv",
                     "tools_over_time.csv", "tools_and_languages_over_time.html"):
            assert (out / name).exists()
        assert "2024" in (out / "tools_and_languages_summary.md").read_text()

    def test_language_trends_run(self, extracted_dir, tmp_path):
        out = tmp_path / "plots"
        language_trends.run(extracted_text_dir=extracted_dir, output_dir=out)
        for name in ("language_usage_by_year.csv", "language_usage_long.csv",
                     "language_trends.html", "language_statistics.txt"):
            assert (out / name).exists()

    def test_repo_extractor_run(self, extracted_dir, tmp_path):
        library = tmp_path / "library.json"
        library.write_text(json.dumps({"entries": {
            "smith2024deep": {"title": "Deep RL", "authors": ["J. Smith"], "year": 2024},
        }}), encoding="utf-8")
        repo_extractor.run(extracted_text_dir=extracted_dir, library_file=library,
                           output_dir=tmp_path)
        report = (tmp_path / "github_repositories.md").read_text(encoding="utf-8")
        assert "https://github.com/example/deep-rl" in report
        assert "**Paper:** Deep RL" in report

    def test_run_comparison_run(self, extracted_dir, tmp_path):
        out = tmp_path / "plots"
        out.mkdir()
        (out / "tools_and_languages_summary.md").write_text(
            "**Analysis Date:** 2025-01-01\n\n**Total Papers Analyzed:** 0\n",
            encoding="utf-8",
        )
        run_comparison.run(extracted_text_dir=extracted_dir, output_dir=out)
        report = (out / "analysis_comparison.md").read_text()
        assert "+1 new papers" in report

    def test_empty_data_exits(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit) as exc:
            specific_tools.run(extracted_text_dir=empty, output_dir=tmp_path)
        assert exc.value.code == 1


class TestSharedPatternTables:
    """The GitHub-URL pattern and pattern tables live once in common."""

    def test_github_pattern_single_source(self):
        assert common.github_pattern.search("https://github.com/user/repo")
        assert common.github_pattern.search("github.com/user/repo")

    def test_specific_tools_tables_are_distinct_data(self):
        # The two specific_tools tables are intentionally different datasets.
        assert "Django" in specific_tools.specific_tools
        assert "Django" not in over_time.specific_tools
