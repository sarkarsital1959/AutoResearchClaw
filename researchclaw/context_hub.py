"""Context Hub (chub) integration for AutoResearchClaw.

Provides curated, versioned API documentation to the code generation and
iterative refinement stages via Andrew Ng's Context Hub CLI.

Instead of the LLM hallucinating API signatures from stale training data,
this module fetches current, LLM-optimized docs before code generation and
annotates gotchas discovered during iterative repair.

Requires: ``npm install -g @aisuite/chub``

Integration points:
  - Stage 10 (CODE_GENERATION): ``fetch_docs_for_topic()`` detects relevant
    libraries and fetches their docs via ``chub get``.
  - Stage 13 (ITERATIVE_REFINE): ``annotate_gotcha()`` saves discovered
    API workarounds so future runs start smarter.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextHubConfig:
    """Configuration for the Context Hub integration.

    Attributes
    ----------
    enabled : bool
        Master switch.  When False, all chub calls are skipped.
    auto_fetch : bool
        Automatically detect and fetch docs during code generation.
    auto_annotate : bool
        Automatically annotate API gotchas discovered during iterative repair.
    max_docs : int
        Maximum number of docs to fetch per stage invocation.
    max_chars : int
        Truncate fetched documentation to this many characters to avoid
        overwhelming the LLM context window.
    lang : str
        Default language variant for ``chub get --lang``.
    timeout_sec : int
        Subprocess timeout for each ``chub`` invocation.
    """

    enabled: bool = False
    auto_fetch: bool = True
    auto_annotate: bool = True
    max_docs: int = 5
    max_chars: int = 12000
    lang: str = "py"
    timeout_sec: int = 30


# ---------------------------------------------------------------------------
# Library detection — maps keywords in topic/plan to chub doc IDs
# ---------------------------------------------------------------------------

# Each entry: (chub_doc_id, [keyword patterns for detection])
# These are the most common libraries an autonomous research pipeline would use.
# The list is intentionally broader than the static framework_docs/ directory
# because chub has a growing community-maintained registry.
_LIBRARY_REGISTRY: list[tuple[str, list[str]]] = [
    # Deep learning
    ("pytorch/api", ["torch", "pytorch", "neural network", "deep learning",
                     "cnn", "rnn", "lstm", "transformer"]),
    ("openai/chat", ["openai", "gpt-4", "gpt-5", "chatgpt", "chat completion"]),
    ("anthropic/sdk", ["anthropic", "claude", "sonnet", "opus"]),
    ("huggingface/transformers", ["huggingface", "transformers", "automodel",
                                   "tokenizer", "bert", "roberta", "llama",
                                   "mistral", "qwen", "gemma"]),
    ("huggingface/datasets", ["datasets", "load_dataset", "huggingface datasets"]),
    # ML / data science
    ("sklearn/api", ["sklearn", "scikit-learn", "random forest", "svm",
                      "logistic regression", "classification", "clustering"]),
    ("numpy/api", ["numpy", "ndarray", "linear algebra", "matrix"]),
    ("pandas/api", ["pandas", "dataframe", "csv", "tabular"]),
    ("scipy/api", ["scipy", "statistical test", "optimization", "signal processing"]),
    ("matplotlib/api", ["matplotlib", "pyplot", "plot", "visualization", "chart"]),
    # RL
    ("gymnasium/api", ["gymnasium", "gym", "openai gym", "reinforcement learning",
                        "rl environment", "mujoco"]),
    # Other
    ("wandb/api", ["wandb", "weights and biases", "experiment tracking"]),
    ("ray/api", ["ray", "ray tune", "distributed training"]),
]


def _detect_libraries(topic: str, hypothesis: str, plan: str) -> list[str]:
    """Detect which chub doc IDs are relevant based on topic/hypothesis/plan.

    Returns up to ``max_docs`` doc IDs sorted by relevance (number of keyword hits).
    """
    combined = (topic + " " + hypothesis + " " + plan).lower()
    scores: dict[str, int] = {}
    for doc_id, keywords in _LIBRARY_REGISTRY:
        hits = sum(1 for kw in keywords if kw in combined)
        if hits > 0:
            scores[doc_id] = hits
    # Sort by hit count descending
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------------------
# chub CLI wrapper
# ---------------------------------------------------------------------------


def _chub_available() -> bool:
    """Check if the ``chub`` CLI is installed and on PATH."""
    return shutil.which("chub") is not None


def _run_chub(args: list[str], timeout_sec: int = 30) -> tuple[int, str, str]:
    """Run a chub CLI command and return (returncode, stdout, stderr)."""
    cmd = ["chub"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.warning("chub command timed out after %ds: %s", timeout_sec, cmd)
        return 1, "", f"Timeout after {timeout_sec}s"
    except FileNotFoundError:
        logger.warning("chub CLI not found — install with: npm install -g @aisuite/chub")
        return 127, "", "chub not found"
    except Exception as exc:
        logger.warning("chub invocation failed: %s", exc)
        return 1, "", str(exc)


def search_docs(query: str, *, limit: int = 10, timeout_sec: int = 30) -> list[dict[str, Any]]:
    """Search the Context Hub registry.

    Returns a list of dicts with at least ``id`` and ``description`` keys.
    """
    rc, stdout, stderr = _run_chub(
        ["search", query, "--json", "--limit", str(limit)],
        timeout_sec=timeout_sec,
    )
    if rc != 0:
        logger.debug("chub search failed (rc=%d): %s", rc, stderr)
        return []
    try:
        data = json.loads(stdout)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "results" in data:
            return data["results"]
    except (json.JSONDecodeError, KeyError):
        pass
    return []


def get_doc(
    doc_id: str,
    *,
    lang: str = "py",
    timeout_sec: int = 30,
) -> str:
    """Fetch a single doc from Context Hub by ID.

    Returns the markdown content, or empty string on failure.
    """
    args = ["get", doc_id, "--lang", lang]
    rc, stdout, stderr = _run_chub(args, timeout_sec=timeout_sec)
    if rc != 0:
        # Try without --lang (some docs have only one variant)
        rc, stdout, stderr = _run_chub(
            ["get", doc_id], timeout_sec=timeout_sec,
        )
    if rc != 0:
        logger.debug("chub get %s failed (rc=%d): %s", doc_id, rc, stderr)
        return ""
    return stdout.strip()


def annotate_doc(
    doc_id: str,
    note: str,
    *,
    timeout_sec: int = 15,
) -> bool:
    """Annotate a doc with a note (persists across sessions).

    Returns True on success.
    """
    rc, _, stderr = _run_chub(
        ["annotate", doc_id, note],
        timeout_sec=timeout_sec,
    )
    if rc != 0:
        logger.debug("chub annotate %s failed: %s", doc_id, stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# High-level integration functions
# ---------------------------------------------------------------------------


def fetch_docs_for_topic(
    topic: str,
    hypothesis: str = "",
    plan: str = "",
    *,
    config: ContextHubConfig | None = None,
) -> str:
    """Detect relevant libraries and fetch their Context Hub docs.

    Returns a formatted string ready to inject into the code generation prompt,
    or empty string if chub is unavailable or nothing was found.

    This is the primary integration point for Stage 10 (CODE_GENERATION).
    """
    cfg = config or ContextHubConfig()
    if not cfg.enabled or not cfg.auto_fetch:
        return ""
    if not _chub_available():
        logger.info("Context Hub: chub CLI not installed, skipping doc fetch")
        return ""

    # Detect relevant libraries
    doc_ids = _detect_libraries(topic, hypothesis, plan)
    if not doc_ids:
        logger.debug("Context Hub: no matching libraries detected")
        return ""
    doc_ids = doc_ids[: cfg.max_docs]

    # Fetch docs
    parts: list[str] = []
    total_chars = 0
    fetched_ids: list[str] = []

    for doc_id in doc_ids:
        if total_chars >= cfg.max_chars:
            break

        content = get_doc(doc_id, lang=cfg.lang, timeout_sec=cfg.timeout_sec)
        if not content:
            # Try searching for it — the doc_id might not match exactly
            results = search_docs(doc_id.split("/")[0], limit=3, timeout_sec=cfg.timeout_sec)
            if results:
                alt_id = results[0].get("id", "")
                if alt_id and alt_id != doc_id:
                    content = get_doc(alt_id, lang=cfg.lang, timeout_sec=cfg.timeout_sec)
                    if content:
                        doc_id = alt_id

        if not content:
            continue

        remaining = cfg.max_chars - total_chars
        if len(content) > remaining:
            content = content[:remaining] + "\n... (truncated)\n"

        parts.append(f"### {doc_id}\n\n{content}")
        total_chars += len(content)
        fetched_ids.append(doc_id)

    if not parts:
        return ""

    logger.info("Context Hub: fetched docs for %s (%d chars)", fetched_ids, total_chars)

    header = (
        "\n## Context Hub — Current API Documentation\n"
        "The following API references were fetched from Context Hub (chub) and "
        "reflect the CURRENT, up-to-date API. Use these exact APIs and patterns — "
        "do NOT rely on memorized API shapes.\n\n"
    )
    return header + "\n---\n\n".join(parts)


def annotate_gotcha(
    stderr: str,
    code: str,
    topic: str = "",
    *,
    config: ContextHubConfig | None = None,
) -> list[str]:
    """Analyze a runtime error and annotate relevant Context Hub docs.

    Looks for import-related errors and common API misuse patterns in stderr,
    then annotates the corresponding chub doc with the gotcha.

    Returns list of doc_ids that were annotated.

    This is the integration point for Stage 13 (ITERATIVE_REFINE).
    """
    cfg = config or ContextHubConfig()
    if not cfg.enabled or not cfg.auto_annotate:
        return []
    if not _chub_available():
        return []
    if not stderr:
        return []

    annotated: list[str] = []

    # Extract library names from import errors
    # Pattern: "ModuleNotFoundError: No module named 'xxx'"
    module_errors = re.findall(
        r"(?:ModuleNotFoundError|ImportError).*?['\"](\w+)['\"]", stderr,
    )

    # Pattern: "AttributeError: module 'xxx' has no attribute 'yyy'"
    attr_errors = re.findall(
        r"AttributeError.*?module\s+['\"](\w+)['\"].*?has no attribute\s+['\"](\w+)['\"]",
        stderr,
    )

    # Pattern: "TypeError: xxx() got an unexpected keyword argument 'yyy'"
    kwarg_errors = re.findall(
        r"TypeError.*?(\w+)\(\).*?unexpected keyword argument\s+['\"](\w+)['\"]",
        stderr,
    )

    # Annotate module errors
    for module_name in module_errors:
        for doc_id, keywords in _LIBRARY_REGISTRY:
            if module_name.lower() in [kw.lower() for kw in keywords]:
                note = f"Import error: module '{module_name}' not available in sandbox"
                if annotate_doc(doc_id, note, timeout_sec=cfg.timeout_sec):
                    annotated.append(doc_id)
                break

    # Annotate attribute errors
    for module_name, attr_name in attr_errors:
        for doc_id, keywords in _LIBRARY_REGISTRY:
            if module_name.lower() in [kw.lower() for kw in keywords]:
                note = f"API change: '{module_name}' has no attribute '{attr_name}'"
                if annotate_doc(doc_id, note, timeout_sec=cfg.timeout_sec):
                    annotated.append(doc_id)
                break

    # Annotate unexpected kwarg errors
    for func_name, kwarg_name in kwarg_errors:
        # Try to match function to a library from imports in the code
        for doc_id, keywords in _LIBRARY_REGISTRY:
            lib_name = doc_id.split("/")[0]
            if lib_name.lower() in code.lower():
                note = (
                    f"API change: {func_name}() does not accept "
                    f"keyword argument '{kwarg_name}'"
                )
                if annotate_doc(doc_id, note, timeout_sec=cfg.timeout_sec):
                    annotated.append(doc_id)
                break

    if annotated:
        logger.info("Context Hub: annotated gotchas on %s", annotated)

    return annotated
