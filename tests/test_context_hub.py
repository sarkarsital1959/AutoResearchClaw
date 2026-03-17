"""Tests for the Context Hub (chub) integration module."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import json
import pytest

from researchclaw.context_hub import (
    ContextHubConfig,
    _detect_libraries,
    _chub_available,
    _run_chub,
    search_docs,
    get_doc,
    annotate_doc,
    fetch_docs_for_topic,
    annotate_gotcha,
)


# ---------------------------------------------------------------------------
# Library detection
# ---------------------------------------------------------------------------


class TestDetectLibraries:
    def test_detects_pytorch(self):
        ids = _detect_libraries("deep learning with PyTorch", "", "")
        assert "pytorch/api" in ids

    def test_detects_sklearn(self):
        ids = _detect_libraries("random forest classification", "", "")
        assert "sklearn/api" in ids

    def test_detects_multiple(self):
        ids = _detect_libraries(
            "transformer model training with huggingface",
            "fine-tune BERT on classification",
            "use sklearn for baselines and matplotlib for plots",
        )
        assert "huggingface/transformers" in ids
        assert "sklearn/api" in ids
        assert "matplotlib/api" in ids

    def test_no_match(self):
        ids = _detect_libraries("the meaning of life", "", "")
        assert ids == []

    def test_sorted_by_hits(self):
        ids = _detect_libraries(
            "deep learning neural network pytorch transformer CNN",
            "", "",
        )
        # pytorch/api should rank highly since it matches many keywords
        assert ids[0] == "pytorch/api"

    def test_hypothesis_and_plan_included(self):
        ids = _detect_libraries(
            "research topic",
            "we will use numpy for matrix operations",
            "implement with scipy optimization",
        )
        assert "numpy/api" in ids
        assert "scipy/api" in ids


# ---------------------------------------------------------------------------
# CLI wrapper (mocked)
# ---------------------------------------------------------------------------


class TestRunChub:
    @patch("shutil.which", return_value="/usr/local/bin/chub")
    def test_chub_available(self, mock_which):
        assert _chub_available() is True

    @patch("shutil.which", return_value=None)
    def test_chub_not_available(self, mock_which):
        assert _chub_available() is False

    @patch("subprocess.run")
    def test_run_chub_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="result", stderr=""
        )
        rc, stdout, stderr = _run_chub(["search", "pytorch"])
        assert rc == 0
        assert stdout == "result"
        mock_run.assert_called_once()

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_run_chub_not_found(self, mock_run):
        rc, stdout, stderr = _run_chub(["search", "pytorch"])
        assert rc == 127
        assert "not found" in stderr


class TestSearchDocs:
    @patch("researchclaw.context_hub._run_chub")
    def test_search_returns_results(self, mock_run):
        mock_run.return_value = (
            0,
            json.dumps([
                {"id": "pytorch/api", "description": "PyTorch API docs"},
                {"id": "numpy/api", "description": "NumPy API docs"},
            ]),
            "",
        )
        results = search_docs("pytorch")
        assert len(results) == 2
        assert results[0]["id"] == "pytorch/api"

    @patch("researchclaw.context_hub._run_chub")
    def test_search_handles_failure(self, mock_run):
        mock_run.return_value = (1, "", "error")
        results = search_docs("pytorch")
        assert results == []

    @patch("researchclaw.context_hub._run_chub")
    def test_search_handles_dict_response(self, mock_run):
        mock_run.return_value = (
            0,
            json.dumps({"results": [{"id": "test/api"}]}),
            "",
        )
        results = search_docs("test")
        assert len(results) == 1


class TestGetDoc:
    @patch("researchclaw.context_hub._run_chub")
    def test_get_returns_content(self, mock_run):
        mock_run.return_value = (0, "# PyTorch API\n...", "")
        content = get_doc("pytorch/api", lang="py")
        assert "PyTorch" in content

    @patch("researchclaw.context_hub._run_chub")
    def test_get_fallback_without_lang(self, mock_run):
        # First call with --lang fails, second without --lang succeeds
        mock_run.side_effect = [
            (1, "", "no lang variant"),
            (0, "# Doc content", ""),
        ]
        content = get_doc("pytorch/api", lang="py")
        assert "Doc content" in content
        assert mock_run.call_count == 2


class TestAnnotateDoc:
    @patch("researchclaw.context_hub._run_chub")
    def test_annotate_success(self, mock_run):
        mock_run.return_value = (0, "", "")
        assert annotate_doc("pytorch/api", "test note") is True

    @patch("researchclaw.context_hub._run_chub")
    def test_annotate_failure(self, mock_run):
        mock_run.return_value = (1, "", "error")
        assert annotate_doc("pytorch/api", "test note") is False


# ---------------------------------------------------------------------------
# High-level integration
# ---------------------------------------------------------------------------


class TestFetchDocsForTopic:
    def test_disabled_returns_empty(self):
        cfg = ContextHubConfig(enabled=False)
        result = fetch_docs_for_topic("deep learning", config=cfg)
        assert result == ""

    @patch("researchclaw.context_hub._chub_available", return_value=False)
    def test_chub_not_installed_returns_empty(self, mock_avail):
        cfg = ContextHubConfig(enabled=True)
        result = fetch_docs_for_topic("deep learning", config=cfg)
        assert result == ""

    @patch("researchclaw.context_hub.get_doc")
    @patch("researchclaw.context_hub._chub_available", return_value=True)
    def test_fetches_and_formats_docs(self, mock_avail, mock_get):
        mock_get.return_value = "# PyTorch API\nUse torch.nn..."
        cfg = ContextHubConfig(enabled=True, max_docs=2, max_chars=50000)
        result = fetch_docs_for_topic(
            "deep learning with PyTorch", config=cfg,
        )
        assert "Context Hub" in result
        assert "PyTorch" in result
        assert mock_get.called

    @patch("researchclaw.context_hub.get_doc")
    @patch("researchclaw.context_hub._chub_available", return_value=True)
    def test_respects_max_chars(self, mock_avail, mock_get):
        mock_get.return_value = "x" * 20000
        cfg = ContextHubConfig(enabled=True, max_chars=5000)
        result = fetch_docs_for_topic(
            "deep learning with PyTorch transformer", config=cfg,
        )
        # Should truncate
        assert len(result) < 10000

    @patch("researchclaw.context_hub.get_doc", return_value="")
    @patch("researchclaw.context_hub.search_docs", return_value=[])
    @patch("researchclaw.context_hub._chub_available", return_value=True)
    def test_no_docs_found_returns_empty(self, mock_avail, mock_search, mock_get):
        cfg = ContextHubConfig(enabled=True)
        result = fetch_docs_for_topic(
            "deep learning with PyTorch", config=cfg,
        )
        assert result == ""


class TestAnnotateGotcha:
    def test_disabled_returns_empty(self):
        cfg = ContextHubConfig(enabled=False)
        result = annotate_gotcha("ImportError: No module named 'torch'", "import torch", config=cfg)
        assert result == []

    @patch("researchclaw.context_hub.annotate_doc", return_value=True)
    @patch("researchclaw.context_hub._chub_available", return_value=True)
    def test_annotates_import_error(self, mock_avail, mock_annotate):
        cfg = ContextHubConfig(enabled=True)
        result = annotate_gotcha(
            "ModuleNotFoundError: No module named 'torch'",
            "import torch\nimport numpy",
            config=cfg,
        )
        assert len(result) > 0
        mock_annotate.assert_called()

    @patch("researchclaw.context_hub.annotate_doc", return_value=True)
    @patch("researchclaw.context_hub._chub_available", return_value=True)
    def test_annotates_attribute_error(self, mock_avail, mock_annotate):
        cfg = ContextHubConfig(enabled=True)
        result = annotate_gotcha(
            "AttributeError: module 'torch' has no attribute 'compile_mode'",
            "import torch",
            config=cfg,
        )
        assert len(result) > 0

    @patch("researchclaw.context_hub._chub_available", return_value=True)
    def test_no_errors_returns_empty(self, mock_avail):
        cfg = ContextHubConfig(enabled=True)
        result = annotate_gotcha("", "import torch", config=cfg)
        assert result == []

    @patch("researchclaw.context_hub._chub_available", return_value=False)
    def test_chub_not_available_returns_empty(self, mock_avail):
        cfg = ContextHubConfig(enabled=True)
        result = annotate_gotcha(
            "ModuleNotFoundError: No module named 'torch'",
            "import torch",
            config=cfg,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestContextHubConfig:
    def test_defaults(self):
        cfg = ContextHubConfig()
        assert cfg.enabled is False
        assert cfg.auto_fetch is True
        assert cfg.auto_annotate is True
        assert cfg.max_docs == 5
        assert cfg.max_chars == 12000
        assert cfg.lang == "py"
        assert cfg.timeout_sec == 30

    def test_rcconfig_has_context_hub(self):
        """Ensure RCConfig includes context_hub field."""
        from researchclaw.config import RCConfig
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(RCConfig)]
        assert "context_hub" in field_names

    def test_parse_from_dict(self):
        """Ensure context_hub config is parsed from YAML dict."""
        from researchclaw.config import _parse_context_hub_config, ContextHubConfig

        data = {
            "enabled": True,
            "max_docs": 3,
            "lang": "js",
        }
        cfg = _parse_context_hub_config(data)
        assert cfg.enabled is True
        assert cfg.max_docs == 3
        assert cfg.lang == "js"
        # Defaults for unspecified fields
        assert cfg.auto_fetch is True
        assert cfg.timeout_sec == 30
