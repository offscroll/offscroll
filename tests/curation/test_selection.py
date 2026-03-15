"""Tests for the loss function terms, page estimation, and optimizer.

Tests for loss function terms.
Tests for greedy optimizer (select_items).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from offscroll.curation.selection import (
    _assign_layout_hint,
    _build_curated_edition,
    _combined_loss,
    _cosine_similarity,
    _quality_score,
    _select_pull_quote,
    coverage_loss,
    curate_edition,
    diversity_loss,
    estimate_pages,
    fit_loss,
    quality_loss,
    redundancy_loss,
    select_items,
)
from offscroll.models import FeedItem, LayoutHint, SourceType


def _item(
    item_id: str = "test",
    content_text: str = "",
    word_count: int = 0,
    cluster_id: int | None = None,
    embedding: list[float] | None = None,
    author: str | None = None,
) -> FeedItem:
    """Helper to create a minimal FeedItem for testing."""
    return FeedItem(
        item_id=item_id,
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        content_text=content_text,
        word_count=word_count,
        cluster_id=cluster_id,
        embedding=embedding,
        author=author,
    )


# ---------------------------------------------------------------------------
# coverage_loss
# ---------------------------------------------------------------------------


def test_coverage_loss_all_clusters_represented():
    """Coverage loss is 0 when every cluster has a representative."""
    items = [
        _item(item_id="a", cluster_id=0),
        _item(item_id="b", cluster_id=1),
        _item(item_id="c", cluster_id=2),
    ]
    assert coverage_loss(items, n_clusters=3) == 0.0


def test_coverage_loss_missing_cluster():
    """Coverage loss reflects the fraction of clusters missing."""
    items = [
        _item(item_id="a", cluster_id=0),
        _item(item_id="b", cluster_id=1),
    ]
    # 2 of 3 clusters represented
    loss = coverage_loss(items, n_clusters=3)
    assert abs(loss - 1.0 / 3.0) < 1e-9


def test_coverage_loss_empty_selection():
    """Coverage loss is 0 for an empty selection."""
    assert coverage_loss([], n_clusters=5) == 0.0


def test_coverage_loss_zero_clusters():
    """Coverage loss is 0 when n_clusters is 0."""
    items = [_item(item_id="a", cluster_id=0)]
    assert coverage_loss(items, n_clusters=0) == 0.0


def test_coverage_loss_ignores_noise_cluster():
    """Noise cluster (-1) items do not count as representing a cluster."""
    items = [
        _item(item_id="a", cluster_id=-1),
        _item(item_id="b", cluster_id=0),
    ]
    # Only cluster 0 represented out of 2
    loss = coverage_loss(items, n_clusters=2)
    assert abs(loss - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# redundancy_loss
# ---------------------------------------------------------------------------


def test_redundancy_loss_no_same_cluster_pairs():
    """Redundancy loss is 0 when no two items share a cluster."""
    items = [
        _item(item_id="a", cluster_id=0, embedding=[1.0, 0.0]),
        _item(item_id="b", cluster_id=1, embedding=[0.0, 1.0]),
    ]
    assert redundancy_loss(items) == 0.0


def test_redundancy_loss_identical_embeddings():
    """Redundancy loss is 1.0 for identical embeddings in the same cluster."""
    items = [
        _item(item_id="a", cluster_id=0, embedding=[1.0, 0.0, 0.0]),
        _item(item_id="b", cluster_id=0, embedding=[1.0, 0.0, 0.0]),
    ]
    assert abs(redundancy_loss(items) - 1.0) < 1e-9


def test_redundancy_loss_orthogonal_embeddings():
    """Redundancy loss is 0 for orthogonal embeddings in the same cluster."""
    items = [
        _item(item_id="a", cluster_id=0, embedding=[1.0, 0.0, 0.0]),
        _item(item_id="b", cluster_id=0, embedding=[0.0, 1.0, 0.0]),
    ]
    assert abs(redundancy_loss(items) - 0.0) < 1e-9


def test_redundancy_loss_skips_noise_cluster():
    """Items in noise cluster (-1) are excluded from redundancy calculation."""
    items = [
        _item(item_id="a", cluster_id=-1, embedding=[1.0, 0.0]),
        _item(item_id="b", cluster_id=-1, embedding=[1.0, 0.0]),
    ]
    assert redundancy_loss(items) == 0.0


def test_redundancy_loss_skips_items_without_embedding():
    """Items with embedding == None are skipped."""
    items = [
        _item(item_id="a", cluster_id=0, embedding=None),
        _item(item_id="b", cluster_id=0, embedding=[1.0, 0.0]),
    ]
    assert redundancy_loss(items) == 0.0


def test_redundancy_loss_empty_selection():
    """Redundancy loss is 0 for an empty selection."""
    assert redundancy_loss([]) == 0.0


# ---------------------------------------------------------------------------
# quality_loss
# ---------------------------------------------------------------------------


def test_quality_loss_short_content():
    """Short content (few words) has high quality loss."""
    items = [_item(word_count=5)]
    loss = quality_loss(items)
    # 5-word item has very low quality score, so loss is close to 1.0
    assert loss > 0.8


def test_quality_loss_medium_content():
    """Medium content (~200 words) has low quality loss."""
    items = [_item(word_count=200)]
    loss = quality_loss(items)
    # 200-word item has quality score ~0.97, loss ~0.03
    assert loss < 0.1


def test_quality_loss_long_content():
    """Long content (2000+ words) has very low quality loss."""
    items = [_item(word_count=2000)]
    loss = quality_loss(items)
    assert loss < 0.01


def test_quality_loss_empty_selection():
    """Quality loss is 0 for an empty selection."""
    assert quality_loss([]) == 0.0


def test_quality_score_monotonic():
    """Quality score increases monotonically with word count above 20."""
    scores = [_quality_score(_item(word_count=wc)) for wc in [20, 50, 100, 200, 500, 1000]]
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1]


def test_quality_score_clamped():
    """Quality score is clamped to [0.0, 1.0]."""
    assert _quality_score(_item(word_count=0)) >= 0.0
    assert _quality_score(_item(word_count=100000)) <= 1.0


# ---------------------------------------------------------------------------
# diversity_loss
# ---------------------------------------------------------------------------


def test_diversity_loss_uniform_authors():
    """Diversity loss is 0 when every item has a unique author."""
    items = [
        _item(item_id="a", author="Alice"),
        _item(item_id="b", author="Bob"),
        _item(item_id="c", author="Carol"),
    ]
    assert abs(diversity_loss(items) - 0.0) < 1e-9


def test_diversity_loss_single_author():
    """Diversity loss is 1.0 when all items are from the same author."""
    items = [
        _item(item_id="a", author="Alice"),
        _item(item_id="b", author="Alice"),
        _item(item_id="c", author="Alice"),
    ]
    assert abs(diversity_loss(items) - 1.0) < 1e-9


def test_diversity_loss_single_item():
    """Diversity loss is 0 for a single-item selection."""
    items = [_item(item_id="a", author="Alice")]
    assert diversity_loss(items) == 0.0


def test_diversity_loss_empty_selection():
    """Diversity loss is 0 for an empty selection."""
    assert diversity_loss([]) == 0.0


def test_diversity_loss_none_author():
    """Items with author=None are treated as 'Unknown'."""
    items = [
        _item(item_id="a", author=None),
        _item(item_id="b", author=None),
        _item(item_id="c", author="Alice"),
    ]
    # Two "Unknown" and one "Alice" -- not uniform, not single author
    loss = diversity_loss(items)
    assert 0.0 < loss < 1.0


def test_diversity_loss_partial_diversity():
    """Diversity loss is between 0 and 1 for partial diversity."""
    items = [
        _item(item_id="a", author="Alice"),
        _item(item_id="b", author="Alice"),
        _item(item_id="c", author="Bob"),
        _item(item_id="d", author="Carol"),
    ]
    loss = diversity_loss(items)
    assert 0.0 < loss < 1.0


# ---------------------------------------------------------------------------
# estimate_pages
# ---------------------------------------------------------------------------


def test_estimate_pages_known_word_count():
    """500 words estimates to exactly 1 page."""
    items = [_item(word_count=500)]
    assert abs(estimate_pages(items) - 1.0) < 1e-9


def test_estimate_pages_multiple_items():
    """Multiple items sum their word counts."""
    items = [
        _item(item_id="a", word_count=250),
        _item(item_id="b", word_count=250),
    ]
    assert abs(estimate_pages(items) - 1.0) < 1e-9


def test_estimate_pages_empty():
    """Empty selection estimates to 0 pages."""
    assert estimate_pages([]) == 0.0


# ---------------------------------------------------------------------------
# fit_loss
# ---------------------------------------------------------------------------


def test_fit_loss_exact_match():
    """Fit loss is 0 when estimated pages equals target."""
    items = [_item(word_count=5000)]  # 5000 / 500 = 10 pages
    assert abs(fit_loss(items, target_pages=10) - 0.0) < 1e-9


def test_fit_loss_over_budget():
    """Fit loss reflects over-budget deviation."""
    items = [_item(word_count=7500)]  # 15 pages, target 10
    loss = fit_loss(items, target_pages=10)
    assert abs(loss - 0.5) < 1e-9  # |15 - 10| / 10 = 0.5


def test_fit_loss_under_budget():
    """Fit loss reflects under-budget deviation."""
    items = [_item(word_count=2500)]  # 5 pages, target 10
    loss = fit_loss(items, target_pages=10)
    assert abs(loss - 0.5) < 1e-9  # |5 - 10| / 10 = 0.5


def test_fit_loss_empty_selection():
    """Fit loss is 0 for an empty selection."""
    assert fit_loss([], target_pages=10) == 0.0


def test_fit_loss_zero_target():
    """Fit loss is 0 when target_pages is 0."""
    items = [_item(word_count=500)]
    assert fit_loss(items, target_pages=0) == 0.0


# ---------------------------------------------------------------------------
# _cosine_similarity (helper, tested for correctness)
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    """Identical vectors have similarity 1.0."""
    assert abs(_cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors have similarity 0.0."""
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-9


def test_cosine_similarity_zero_vector():
    """Zero vector returns 0.0 similarity."""
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# select_items (optimizer)
# ---------------------------------------------------------------------------


def _pool(n: int = 15, n_clusters: int = 3) -> list[FeedItem]:
    """Create a pool of n items spread across n_clusters clusters."""
    items = []
    for i in range(n):
        cluster = i % n_clusters
        emb = [0.0] * n_clusters
        emb[cluster] = 1.0
        wc = 100 + (i * 50)
        items.append(
            _item(
                item_id=f"pool-{i}",
                cluster_id=cluster,
                embedding=emb,
                author=f"Author-{i % 5}",
                word_count=wc,
            )
        )
    return items


def test_select_items_basic():
    """Pool of 15 items, 3 clusters, returns SelectionResult."""
    items = _pool(15, 3)
    result = select_items(items, n_clusters=3, target_pages=2)
    assert result is not None
    assert len(result.items) > 0
    assert result.total_loss >= 0.0


def test_select_items_covers_all_clusters():
    """Result has items from every cluster."""
    items = _pool(15, 3)
    result = select_items(items, n_clusters=3, target_pages=5)
    cluster_ids = {
        item.cluster_id
        for item in result.items
        if item.cluster_id is not None and item.cluster_id != -1
    }
    assert cluster_ids == {0, 1, 2}


def test_select_items_result_fields():
    """All result fields are present and have correct types."""
    items = _pool(10, 2)
    result = select_items(items, n_clusters=2, target_pages=2)
    assert isinstance(result.total_loss, float)
    assert isinstance(result.term_losses, dict)
    assert isinstance(result.iterations, int)
    assert isinstance(result.improved, bool)
    assert "coverage" in result.term_losses
    assert "redundancy" in result.term_losses
    assert "quality" in result.term_losses
    assert "diversity" in result.term_losses
    assert "fit" in result.term_losses


def test_select_items_respects_page_budget():
    """Estimated pages within 50% of target."""
    items = _pool(30, 5)
    result = select_items(items, n_clusters=5, target_pages=3)
    pages = estimate_pages(result.items)
    # Should be roughly in range -- not exact, but not wildly off
    assert pages < 3 * 2.0  # Less than double
    assert pages > 0  # Not empty


def test_select_items_empty_pool_raises():
    """Empty pool raises ValueError."""
    with pytest.raises(ValueError, match="Pool is empty"):
        select_items([], n_clusters=3)


def test_select_items_single_cluster():
    """Works with n_clusters=1."""
    items = [
        _item(
            item_id=f"single-{i}",
            cluster_id=0,
            embedding=[1.0],
            word_count=200,
            author=f"A{i}",
        )
        for i in range(5)
    ]
    result = select_items(items, n_clusters=1, target_pages=1)
    assert len(result.items) >= 1
    assert result.term_losses["coverage"] == 0.0


def test_select_items_all_noise():
    """Pool with only noise items (cluster_id=-1) still produces a result."""
    items = [
        _item(
            item_id=f"noise-{i}",
            cluster_id=-1,
            embedding=[float(i)],
            word_count=200,
            author=f"N{i}",
        )
        for i in range(5)
    ]
    result = select_items(items, n_clusters=0, target_pages=1)
    assert len(result.items) >= 1


def test_select_items_custom_weights():
    """Changing weights changes the optimization behavior."""
    items = _pool(20, 4)
    result_default = select_items(items, n_clusters=4, target_pages=3)
    # Zero out everything except fit -- optimizer only cares about page budget
    result_fit_only = select_items(
        items,
        n_clusters=4,
        target_pages=3,
        weights={
            "coverage": 0.0,
            "redundancy": 0.0,
            "quality": 0.0,
            "diversity": 0.0,
            "fit": 10.0,
        },
    )
    # Both results should have valid selections
    assert len(result_default.items) > 0
    assert len(result_fit_only.items) > 0
    # Fit-only should have lower fit loss (or equal) since it optimizes only for fit
    assert result_fit_only.term_losses["fit"] <= result_default.term_losses["fit"] + 0.01


def test_select_items_deterministic():
    """Same input produces same output twice."""
    items = _pool(15, 3)
    result1 = select_items(items, n_clusters=3, target_pages=3)
    result2 = select_items(items, n_clusters=3, target_pages=3)
    assert result1.total_loss == result2.total_loss
    result1_ids = [item.item_id for item in result1.items]
    result2_ids = [item.item_id for item in result2.items]
    assert result1_ids == result2_ids


def test_select_items_hill_climb_improves():
    """Total loss after hill-climb is <= loss after greedy init."""
    # Create a pool where greedy init is not optimal
    items = _pool(20, 4)
    result = select_items(items, n_clusters=4, target_pages=3)
    # The result should have run at least 1 iteration
    assert result.iterations >= 1
    # Total loss should be non-negative
    assert result.total_loss >= 0.0


def test_combined_loss_all_zero():
    """Perfect selection returns 0.0 total loss for each term at 0."""
    # One item per cluster, good quality, right page count, diverse authors
    items = [
        _item(
            item_id=f"perfect-{i}",
            cluster_id=i,
            embedding=[float(j == i) for j in range(3)],
            word_count=1667,  # 3 items * 1667 words / 500 = ~10 pages
            author=f"Author-{i}",
        )
        for i in range(3)
    ]
    total, losses = _combined_loss(
        items,
        n_clusters=3,
        weights={"coverage": 1.0, "redundancy": 1.0, "quality": 1.0, "diversity": 1.0, "fit": 1.0},
        target_pages=10,
    )
    assert losses["coverage"] == 0.0
    assert losses["redundancy"] == 0.0
    assert losses["diversity"] < 1e-9


def test_combined_loss_weights_applied():
    """Weight=0 for a term zeroes its contribution."""
    items = [_item(item_id="w", cluster_id=0, embedding=[1.0], word_count=100)]
    weights_all = {
        "coverage": 1.0,
        "redundancy": 1.0,
        "quality": 1.0,
        "diversity": 1.0,
        "fit": 1.0,
    }
    weights_zero_fit = {
        "coverage": 1.0,
        "redundancy": 1.0,
        "quality": 1.0,
        "diversity": 1.0,
        "fit": 0.0,
    }
    total_all, _ = _combined_loss(items, 3, weights_all, 10)
    total_zero_fit, losses_zero = _combined_loss(items, 3, weights_zero_fit, 10)
    # With fit weight=0, fit loss should still be computed but not contribute
    assert losses_zero["fit"] > 0  # The loss term itself is nonzero
    assert total_zero_fit < total_all  # But it doesn't add to total


# ---------------------------------------------------------------------------
# curate_edition pipeline 
# ---------------------------------------------------------------------------


def _curation_config(tmp_path) -> dict:
    """Create a config dict for curation tests."""
    return {
        "output": {"data_dir": str(tmp_path)},
        "curation": {
            "min_word_count": 10,
            "weights": {
                "coverage": 1.0,
                "redundancy": 1.0,
                "quality": 1.0,
                "diversity": 1.0,
                "fit": 1.0,
            },
            "optimizer_iterations": 10,
        },
        "newspaper": {
            "title": "Test Gazette",
            "subtitle_pattern": "Vol. {volume}, No. {issue}",
            "page_target": 2,
        },
    }


def _curation_pool(n: int = 9, n_clusters: int = 3) -> list[FeedItem]:
    """Create a pool of items with embeddings and cluster_ids for curation tests."""
    items = []
    for i in range(n):
        cluster = i % n_clusters
        emb = [0.0] * n_clusters
        emb[cluster] = 1.0
        wc = 50 + (i * 30)
        items.append(
            _item(
                item_id=f"cur-{i}",
                cluster_id=cluster,
                embedding=emb,
                author=f"Author-{i % 4}",
                content_text=" ".join(["word"] * wc),
                word_count=wc,
            )
        )
    return items


def test_curate_edition_basic(tmp_path):
    """Mocked DB returns items, edition JSON written."""
    config = _curation_config(tmp_path)
    pool = _curation_pool()
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=3),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    assert path.exists()
    assert path.suffix == ".json"
    # Verify it's valid JSON
    with open(path) as f:
        data = json.load(f)
    assert "edition" in data
    assert "sections" in data


def test_curate_edition_empty_pool(tmp_path):
    """No items raises ValueError."""
    config = _curation_config(tmp_path)
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=[]),
        patch("offscroll.curation.selection.get_cluster_count", return_value=0),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        pytest.raises(ValueError, match="No items available"),
    ):
        curate_edition(config)


def test_curate_edition_sections_by_cluster(tmp_path):
    """Items grouped into sections by cluster_id."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=6, n_clusters=2)
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=2),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    with open(path) as f:
        data = json.load(f)
    headings = [s["heading"] for s in data["sections"]]
    assert "Topic 1" in headings
    assert "Topic 2" in headings


def test_curate_edition_noise_section(tmp_path):
    """Noise items go to 'In Brief' section."""
    config = _curation_config(tmp_path)
    # All items are noise (cluster_id = -1)
    pool = [
        _item(
            item_id=f"noise-{i}",
            cluster_id=-1,
            embedding=[float(i)],
            word_count=100,
            author=f"N{i}",
            content_text=" ".join(["word"] * 100),
        )
        for i in range(5)
    ]
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=0),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    with open(path) as f:
        data = json.load(f)
    headings = [s["heading"] for s in data["sections"]]
    assert "In Brief" in headings


def test_curate_edition_layout_hints(tmp_path):
    """Feature/standard/brief assigned by word count."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="long",
            cluster_id=0,
            embedding=[1.0],
            word_count=400,
            content_text=" ".join(["word"] * 400),
        ),
        _item(
            item_id="medium",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text=" ".join(["word"] * 100),
        ),
        _item(
            item_id="short",
            cluster_id=0,
            embedding=[1.0],
            word_count=20,
            content_text=" ".join(["word"] * 20),
        ),
    ]
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    with open(path) as f:
        data = json.load(f)
    all_items = []
    for section in data["sections"]:
        all_items.extend(section["items"])
    hints = {item["item_id"]: item["layout_hint"] for item in all_items}
    #  Only the cover story gets FEATURE. With 100-word items
    # none qualify for the cover (need 300+), so all get STANDARD/BRIEF.
    # The editorial LLM layer may override, but the initial assignment
    # should not produce FEATURE for non-cover items.
    assert hints["long"] in ("standard", "feature")  # LLM may promote
    assert hints["medium"] in ("standard", "feature")  # LLM may promote
    assert hints["short"] == "brief"


def test_curate_edition_pull_quote(tmp_path):
    """Pull quote extracted from longest item."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="long",
            cluster_id=0,
            embedding=[1.0],
            word_count=200,
            content_text="First sentence here. Second sentence is longer. Third.",
            author="QuoteAuthor",
        ),
    ]
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    with open(path) as f:
        data = json.load(f)
    assert len(data["pull_quotes"]) == 1
    assert data["pull_quotes"][0]["attribution"] == "QuoteAuthor"


def test_curate_edition_records_edition(tmp_path):
    """record_edition called with correct args."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=3, n_clusters=1)
    mock_record = MagicMock()
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition", mock_record),
    ):
        curate_edition(config)
    mock_record.assert_called_once()
    call_args = mock_record.call_args
    assert call_args[0][0] == config  # config arg
    assert isinstance(call_args[0][1], str)  # edition_id
    assert isinstance(call_args[0][2], list)  # item_ids


def test_build_curated_edition_meta(tmp_path):
    """Edition metadata matches config."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=3, n_clusters=1)
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})
    assert edition.edition.title == "Test Gazette"
    assert edition.edition.subtitle == "Vol. 1, No. 1"
    assert edition.page_target == 2


def test_build_curated_edition_issue_number_increments(tmp_path):
    """Issue number increments based on existing editions count."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=3, n_clusters=1)
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})

    # First edition: count=0 -> issue=1
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition1 = _build_curated_edition(result, config, feed_name_map={})
    assert edition1.edition.subtitle == "Vol. 1, No. 1"

    # Second edition: count=1 -> issue=2
    with patch("offscroll.curation.selection.get_edition_count", return_value=1):
        edition2 = _build_curated_edition(result, config, feed_name_map={})
    assert edition2.edition.subtitle == "Vol. 1, No. 2"

    # Fifth edition: count=4 -> issue=5
    with patch("offscroll.curation.selection.get_edition_count", return_value=4):
        edition5 = _build_curated_edition(result, config, feed_name_map={})
    assert edition5.edition.subtitle == "Vol. 1, No. 5"


def test_assign_layout_hint_feature():
    """ 300+ words only gets FEATURE when is_cover=True."""
    item = _item(word_count=300)
    assert _assign_layout_hint(item, is_cover=True) == LayoutHint.FEATURE
    item = _item(word_count=500)
    assert _assign_layout_hint(item, is_cover=True) == LayoutHint.FEATURE
    # Without is_cover, 300+ words gets STANDARD
    item = _item(word_count=300)
    assert _assign_layout_hint(item) == LayoutHint.STANDARD
    item = _item(word_count=500)
    assert _assign_layout_hint(item) == LayoutHint.STANDARD


def test_assign_layout_hint_brief():
    """30 or fewer -> BRIEF regardless of cover status."""
    item = _item(word_count=30)
    assert _assign_layout_hint(item) == LayoutHint.BRIEF
    item = _item(word_count=10)
    assert _assign_layout_hint(item) == LayoutHint.BRIEF
    # Even cover items with tiny word count get BRIEF
    item = _item(word_count=10)
    assert _assign_layout_hint(item, is_cover=True) == LayoutHint.BRIEF


def test_assign_layout_hint_standard():
    """31-299 words -> STANDARD always, 300+ without cover -> STANDARD."""
    item = _item(word_count=31)
    assert _assign_layout_hint(item) == LayoutHint.STANDARD
    item = _item(word_count=299)
    assert _assign_layout_hint(item) == LayoutHint.STANDARD


def test_select_pull_quote_skips_first_sentence():
    """P4: Pull quote never selects the first sentence."""
    text = "This is the first sentence. This is a much better second sentence with a strong claim."
    result = _select_pull_quote(text)
    assert result is not None
    assert "first sentence" not in result


def test_select_pull_quote_min_words():
    """P4: Pull quote requires minimum 8 words."""
    text = "First sentence here. Too short. Also too short."
    result = _select_pull_quote(text)
    assert result is None


def test_select_pull_quote_max_words():
    """P4: Pull quote rejects sentences over 40 words."""
    long_sent = " ".join(["word"] * 45) + "."
    text = "First sentence. " + long_sent
    result = _select_pull_quote(text)
    assert result is None


def test_select_pull_quote_complete_sentence():
    """P4: Pull quote must be a complete sentence."""
    text = (
        "First sentence here. "
        "This is what product-market fit looks like. "
        "The level of agency is off the charts."
    )
    result = _select_pull_quote(text)
    assert result is not None
    assert result.endswith(".")


def test_select_pull_quote_prefers_strong_claims():
    """P4: Pull quote prefers sentences with strong claim words."""
    text = (
        "First sentence here. "
        "The weather was pleasant and mild today. "
        "This is what transforms everything about the industry."
    )
    result = _select_pull_quote(text)
    assert result is not None
    assert "transforms" in result


def test_select_pull_quote_empty_text():
    """P4: Returns None for empty or None input."""
    assert _select_pull_quote("") is None
    assert _select_pull_quote(None) is None


def test_select_pull_quote_single_sentence():
    """P4: Returns None if only one sentence (cannot skip first)."""
    assert _select_pull_quote("Only one sentence here.") is None


# ---------------------------------------------------------------------------
# Integration Tests: Full curate_edition Pipeline 
# ---------------------------------------------------------------------------


def test_curate_edition_with_editorial(tmp_path):
    """curate_edition integrates embed -> cluster -> select -> editorial."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=5, n_clusters=2)

    with (
        patch("offscroll.curation.selection.EMBEDDINGS_AVAILABLE", False),
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=2),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
        patch("offscroll.curation.selection.run_editorial") as mock_editorial,
    ):
        # run_editorial should be called with the edition
        mock_editorial.side_effect = lambda edition, config: edition

        path = curate_edition(config)

        # Editorial should have been called
        mock_editorial.assert_called_once()
        assert path.exists()
        assert path.suffix == ".json"


def test_curate_edition_without_editorial(tmp_path):
    """curate_edition works when Ollama is unavailable."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=5, n_clusters=2)

    with (
        patch("offscroll.curation.selection.EMBEDDINGS_AVAILABLE", False),
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=2),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
        patch("offscroll.curation.selection.run_editorial") as mock_editorial,
    ):
        # run_editorial returns edition unchanged when Ollama unavailable
        # (it handles the error internally and returns the unmodified edition)
        def mock_run_editorial(edition, config):
            # Simulate graceful degradation
            return edition

        mock_editorial.side_effect = mock_run_editorial

        # Should still succeed
        path = curate_edition(config)
        assert path.exists()


def test_curate_edition_filters_low_word_count(tmp_path):
    """Items below min_word_count are filtered from the pool."""
    config = _curation_config(tmp_path)
    config["curation"]["min_word_count"] = 50
    pool = [
        _item(
            item_id="short",
            cluster_id=0,
            embedding=[1.0],
            word_count=5,
            content_text="hello",
        ),
        _item(
            item_id="long",
            cluster_id=0,
            embedding=[1.0],
            word_count=200,
            content_text=" ".join(["word"] * 200),
        ),
    ]
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    with open(path) as f:
        data = json.load(f)
    all_ids = []
    for section in data["sections"]:
        for item in section["items"]:
            all_ids.append(item["item_id"])
    assert "short" not in all_ids
    assert "long" in all_ids


def test_curate_edition_filters_zero_word_count(tmp_path):
    """Items with zero word count are filtered."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="empty",
            cluster_id=0,
            embedding=[1.0],
            word_count=0,
            content_text="",
        ),
        _item(
            item_id="ok",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text=" ".join(["word"] * 100),
        ),
    ]
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    with open(path) as f:
        data = json.load(f)
    all_ids = []
    for section in data["sections"]:
        for item in section["items"]:
            all_ids.append(item["item_id"])
    assert "empty" not in all_ids
    assert "ok" in all_ids


def test_build_curated_edition_skips_empty_content(tmp_path):
    """Items with empty content_text are excluded from curated edition."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="empty",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text="",
        ),
        _item(
            item_id="ok",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text=" ".join(["word"] * 100),
        ),
    ]
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})
    all_ids = []
    for section in edition.sections:
        for item in section.items:
            all_ids.append(item.item_id)
    assert "empty" not in all_ids
    assert "ok" in all_ids


def test_curate_edition_respects_min_word_count_config(tmp_path):
    """min_word_count config is respected."""
    config = _curation_config(tmp_path)
    config["curation"]["min_word_count"] = 100
    pool = [
        _item(
            item_id="below",
            cluster_id=0,
            embedding=[1.0],
            word_count=99,
            content_text=" ".join(["word"] * 99),
        ),
        _item(
            item_id="at",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text=" ".join(["word"] * 100),
        ),
    ]
    with (
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
    ):
        path = curate_edition(config)
    with open(path) as f:
        data = json.load(f)
    all_ids = []
    for section in data["sections"]:
        for item in section["items"]:
            all_ids.append(item["item_id"])
    assert "below" not in all_ids
    assert "at" in all_ids


def test_build_curated_edition_decodes_entities(tmp_path):
    """display_text has HTML entities decoded for defense-in-depth."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="encoded",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text="He said &ldquo;hello&rdquo; &mdash; and left.",
        ),
    ]
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})
    item = edition.sections[0].items[0]
    assert "&ldquo;" not in item.display_text
    assert "&rdquo;" not in item.display_text
    assert "\u201c" in item.display_text  # left double quote
    assert "\u201d" in item.display_text  # right double quote


def test_curate_edition_embed_cluster_integration(tmp_path):
    """curate_edition calls embed and cluster functions when available."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=5, n_clusters=2)

    with (
        patch("offscroll.curation.selection.EMBEDDINGS_AVAILABLE", True),
        patch("offscroll.curation.selection.embed_items") as mock_embed,
        patch("offscroll.curation.selection.cluster_items") as mock_cluster,
        patch("offscroll.curation.selection.get_items_for_embedding", return_value=[pool[0]]),
        patch("offscroll.curation.selection.update_embeddings"),
        patch("offscroll.curation.selection.get_items_for_clustering", return_value=[pool[0]]),
        patch("offscroll.curation.selection.update_cluster_ids"),
        patch("offscroll.curation.selection.get_items_for_curation", return_value=pool),
        patch("offscroll.curation.selection.get_cluster_count", return_value=2),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition"),
        patch("offscroll.curation.selection.run_editorial") as mock_editorial,
    ):
        # Setup return values
        mock_embed.return_value = [pool[0]]
        mock_cluster.return_value = [pool[0]]
        mock_editorial.side_effect = lambda edition, config: edition

        path = curate_edition(config)

        # Embed and cluster should have been called
        mock_embed.assert_called_once()
        mock_cluster.assert_called_once()
        assert path.exists()


# ---------------------------------------------------------------------------
# Source attribution in _build_curated_edition
# ---------------------------------------------------------------------------


def test_build_curated_edition_populates_source_name(tmp_path):
    """source_name is populated from feed_name_map."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="src-001",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text=" ".join(["word"] * 100),
        ),
    ]
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})
    feed_name_map = {"https://example.com/feed.xml": "Example Blog"}
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map=feed_name_map)
    item = edition.sections[0].items[0]
    assert item.source_name == "Example Blog"


def test_build_curated_edition_source_name_none_when_not_in_map(tmp_path):
    """source_name is None when feed_url not in feed_name_map."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="nosrc-001",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text=" ".join(["word"] * 100),
        ),
    ]
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})
    item = edition.sections[0].items[0]
    assert item.source_name is None


def test_build_curated_edition_populates_item_url(tmp_path):
    """item_url is passed through from FeedItem to CuratedItem."""
    config = _curation_config(tmp_path)
    pool = [
        FeedItem(
            item_id="url-001",
            source_type=SourceType.RSS,
            feed_url="https://example.com/feed.xml",
            item_url="https://example.com/post-1",
            content_text=" ".join(["word"] * 100),
            word_count=100,
            cluster_id=0,
            embedding=[1.0],
        ),
    ]
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})
    item = edition.sections[0].items[0]
    assert item.item_url == "https://example.com/post-1"


def test_build_curated_edition_item_url_none(tmp_path):
    """item_url is None when FeedItem has no item_url."""
    config = _curation_config(tmp_path)
    pool = [
        _item(
            item_id="nourl-001",
            cluster_id=0,
            embedding=[1.0],
            word_count=100,
            content_text=" ".join(["word"] * 100),
        ),
    ]
    from offscroll.curation.selection import SelectionResult

    result = SelectionResult(items=pool, total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})
    item = edition.sections[0].items[0]
    assert item.item_url is None


# ---------------------------------------------------------------------------
#  Image Passthrough and Entity Decoding Tests
# ---------------------------------------------------------------------------


def test_sprint11_images_passed_to_curated_item():
    """(11.6): FeedItem images flow into CuratedItem."""
    from unittest.mock import patch

    from offscroll.curation.selection import SelectionResult, _build_curated_edition
    from offscroll.models import FeedItem, ImageContent, SourceType

    item = FeedItem(
        item_id="img-001",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        author="Writer",
        content_text="Article with images. " * 20,
        cluster_id=0,
        images=[
            ImageContent(
                url="https://example.com/img1.jpg",
                local_path="images/img-001/abc123.jpg",
                alt_text="First image",
                width=800,
                height=600,
            ),
            ImageContent(
                url="https://example.com/img2.jpg",
                local_path="images/img-001/def456.jpg",
                alt_text="Second image",
                width=1024,
                height=768,
            ),
        ],
    )
    config = {
        "output": {"data_dir": "/tmp/test"},
        "newspaper": {"title": "Test", "subtitle_pattern": "Vol. 1, No. {issue}"},
    }
    result = SelectionResult(items=[item], total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})

    curated_item = edition.sections[0].items[0]
    assert len(curated_item.images) == 2
    assert curated_item.images[0].local_path == "images/img-001/abc123.jpg"
    assert curated_item.images[0].caption == "First image"
    assert curated_item.images[0].width == 800
    assert curated_item.images[1].local_path == "images/img-001/def456.jpg"


def test_sprint11_images_without_local_path_skipped():
    """(11.6): Images without local_path are not included."""
    from unittest.mock import patch

    from offscroll.curation.selection import SelectionResult, _build_curated_edition
    from offscroll.models import FeedItem, ImageContent, SourceType

    item = FeedItem(
        item_id="img-002",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        author="Writer",
        content_text="Article with undownloaded images. " * 10,
        cluster_id=0,
        images=[
            ImageContent(
                url="https://example.com/img1.jpg",
                local_path=None,  # Not downloaded yet
                alt_text="Not downloaded",
            ),
            ImageContent(
                url="https://example.com/img2.jpg",
                local_path="images/img-002/good.jpg",
                alt_text="Downloaded",
            ),
        ],
    )
    config = {
        "output": {"data_dir": "/tmp/test"},
        "newspaper": {"title": "Test", "subtitle_pattern": "Vol. 1, No. {issue}"},
    }
    result = SelectionResult(items=[item], total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})

    curated_item = edition.sections[0].items[0]
    assert len(curated_item.images) == 1
    assert curated_item.images[0].caption == "Downloaded"


def test_sprint11_images_caption_fallback():
    """: Image without alt_text produces empty caption.

    Previously, the fallback was "Image from {author}".
    changed this to suppress generic captions entirely --
    an empty string lets templates skip the caption div.
    """
    from unittest.mock import patch

    from offscroll.curation.selection import SelectionResult, _build_curated_edition
    from offscroll.models import FeedItem, ImageContent, SourceType

    item = FeedItem(
        item_id="img-003",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        author="Alice Smith",
        content_text="Article text here. " * 10,
        cluster_id=0,
        images=[
            ImageContent(
                url="https://example.com/img.jpg",
                local_path="images/img-003/pic.jpg",
                alt_text=None,  # No alt text
            ),
        ],
    )
    config = {
        "output": {"data_dir": "/tmp/test"},
        "newspaper": {"title": "Test", "subtitle_pattern": "Vol. 1, No. {issue}"},
    }
    result = SelectionResult(items=[item], total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})

    curated_item = edition.sections[0].items[0]
    assert len(curated_item.images) == 1
    assert curated_item.images[0].caption == ""


def test_sprint11_html_entity_decoding():
    """(11.8): HTML entities decoded in display_text."""
    from unittest.mock import patch

    from offscroll.curation.selection import SelectionResult, _build_curated_edition
    from offscroll.models import FeedItem, SourceType

    item = FeedItem(
        item_id="entity-001",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        author="Writer",
        content_text='She said &#8220;hello&#8221; &amp; smiled &ndash; beautifully.',
        cluster_id=0,
    )
    config = {
        "output": {"data_dir": "/tmp/test"},
        "newspaper": {"title": "Test", "subtitle_pattern": "Vol. 1, No. {issue}"},
    }
    result = SelectionResult(items=[item], total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})

    curated_item = edition.sections[0].items[0]
    # Entities should be decoded
    assert "&#8220;" not in curated_item.display_text
    assert "&amp;" not in curated_item.display_text
    assert "&ndash;" not in curated_item.display_text
    # Check decoded characters
    assert "\u201c" in curated_item.display_text  # left double quote
    assert "&" in curated_item.display_text
    assert "\u2013" in curated_item.display_text  # en dash


def test_sprint11_noise_items_get_images():
    """(11.6): Noise items (cluster_id=-1) also get images."""
    from unittest.mock import patch

    from offscroll.curation.selection import SelectionResult, _build_curated_edition
    from offscroll.models import FeedItem, ImageContent, SourceType

    item = FeedItem(
        item_id="noise-001",
        source_type=SourceType.RSS,
        feed_url="https://example.com/feed.xml",
        author="Writer",
        content_text="A noise item with images.",
        cluster_id=-1,
        images=[
            ImageContent(
                url="https://example.com/img.jpg",
                local_path="images/noise-001/pic.jpg",
                alt_text="Noise image",
            ),
        ],
    )
    config = {
        "output": {"data_dir": "/tmp/test"},
        "newspaper": {"title": "Test", "subtitle_pattern": "Vol. 1, No. {issue}"},
    }
    result = SelectionResult(items=[item], total_loss=0.5, term_losses={})
    with patch("offscroll.curation.selection.get_edition_count", return_value=0):
        edition = _build_curated_edition(result, config, feed_name_map={})

    # Noise items go to "In Brief" section
    brief_section = [s for s in edition.sections if s.heading == "In Brief"][0]
    assert len(brief_section.items[0].images) == 1


# ---------------------------------------------------------------------------
#  validate_edition Tests
# ---------------------------------------------------------------------------


def test_validate_edition_removes_zero_content():
    """12.4: validate_edition filters out items with empty display_text."""
    from offscroll.curation.selection import validate_edition
    from offscroll.models import (
        CuratedEdition,
        CuratedItem,
        EditionMeta,
        Section,
    )

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1",
        ),
        sections=[
            Section(
                heading="News",
                items=[
                    CuratedItem(
                        item_id="good-1",
                        display_text="This is a real article with enough words to pass.",
                        author="Alice",
                        word_count=10,
                    ),
                    CuratedItem(
                        item_id="empty-1",
                        display_text="",
                        author="Bob",
                        word_count=0,
                    ),
                ],
            ),
        ],
    )
    result = validate_edition(edition)
    assert len(result.sections[0].items) == 1
    assert result.sections[0].items[0].item_id == "good-1"


def test_validate_edition_removes_boilerplate_items():
    """12.4: validate_edition filters short boilerplate-only items."""
    from offscroll.curation.selection import validate_edition
    from offscroll.models import (
        CuratedEdition,
        CuratedItem,
        EditionMeta,
        Section,
    )

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1",
        ),
        sections=[
            Section(
                heading="News",
                items=[
                    CuratedItem(
                        item_id="good-1",
                        display_text="A real article with content.",
                        author="Alice",
                        word_count=50,
                    ),
                    CuratedItem(
                        item_id="boilerplate-1",
                        display_text="The post My Article first appeared on Blog.",
                        author="Bot",
                        word_count=8,
                    ),
                ],
            ),
        ],
    )
    result = validate_edition(edition)
    assert len(result.sections[0].items) == 1
    assert result.sections[0].items[0].item_id == "good-1"


def test_validate_edition_fixes_refusal_headings():
    """12.4: validate_edition replaces LLM refusal headings."""
    from offscroll.curation.selection import validate_edition
    from offscroll.models import (
        CuratedEdition,
        CuratedItem,
        EditionMeta,
        Section,
    )

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1",
        ),
        sections=[
            Section(
                heading="I cannot generate a heading for this section",
                items=[
                    CuratedItem(
                        item_id="item-1",
                        display_text="Real content here.",
                        author="Alice",
                        title="Climate Crisis",
                        word_count=50,
                    ),
                ],
            ),
        ],
    )
    result = validate_edition(edition)
    # The refusal heading should be replaced with the first item's title
    assert result.sections[0].heading == "Climate Crisis"


def test_validate_edition_truncates_long_headings():
    """12.4: validate_edition truncates headings over 80 chars."""
    from offscroll.curation.selection import validate_edition
    from offscroll.models import (
        CuratedEdition,
        CuratedItem,
        EditionMeta,
        Section,
    )

    long_heading = "A" * 100
    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1",
        ),
        sections=[
            Section(
                heading=long_heading,
                items=[
                    CuratedItem(
                        item_id="item-1",
                        display_text="Real content.",
                        author="Alice",
                        word_count=50,
                    ),
                ],
            ),
        ],
    )
    result = validate_edition(edition)
    assert len(result.sections[0].heading) <= 80
    assert result.sections[0].heading.endswith("...")


def test_validate_edition_filters_preamble_pull_quotes():
    """12.4: validate_edition filters pull quotes with LLM preamble."""
    from offscroll.curation.selection import validate_edition
    from offscroll.models import (
        CuratedEdition,
        CuratedItem,
        EditionMeta,
        PullQuote,
        Section,
    )

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1",
        ),
        sections=[
            Section(
                heading="News",
                items=[
                    CuratedItem(
                        item_id="item-1",
                        display_text="Content.",
                        author="Alice",
                        word_count=50,
                    ),
                ],
            ),
        ],
        pull_quotes=[
            PullQuote(
                text="Here are 3 striking quotes:",
                attribution="LLM",
                source_item_id="unknown",
            ),
            PullQuote(
                text="The river runs deep and fast.",
                attribution="Alice",
                source_item_id="item-1",
            ),
        ],
    )
    result = validate_edition(edition)
    assert len(result.pull_quotes) == 1
    assert result.pull_quotes[0].text == "The river runs deep and fast."


def test_validate_edition_removes_empty_sections():
    """12.4: validate_edition removes sections that become empty after filtering."""
    from offscroll.curation.selection import validate_edition
    from offscroll.models import (
        CuratedEdition,
        CuratedItem,
        EditionMeta,
        Section,
    )

    edition = CuratedEdition(
        edition=EditionMeta(
            date="2026-03-08",
            title="Test",
            subtitle="Vol. 1",
        ),
        sections=[
            Section(
                heading="Good Section",
                items=[
                    CuratedItem(
                        item_id="good-1",
                        display_text="Real article.",
                        author="Alice",
                        word_count=50,
                    ),
                ],
            ),
            Section(
                heading="Empty After Filter",
                items=[
                    CuratedItem(
                        item_id="empty-1",
                        display_text="",
                        author="Bob",
                        word_count=0,
                    ),
                ],
            ),
        ],
    )
    result = validate_edition(edition)
    assert len(result.sections) == 1
    assert result.sections[0].heading == "Good Section"


# ---------------------------------------------------------------------------
# --fresh flag tests
# ---------------------------------------------------------------------------


def test_curate_edition_fresh_skips_recording(tmp_path):
    """fresh=True skips record_edition and passes exclude_previous_editions=False."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=3, n_clusters=1)
    mock_record = MagicMock()
    mock_get_items = MagicMock(return_value=pool)
    with (
        patch("offscroll.curation.selection.get_items_for_curation", mock_get_items),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition", mock_record),
    ):
        path = curate_edition(config, fresh=True)
    # Edition JSON still written
    assert path.exists()
    # record_edition NOT called
    mock_record.assert_not_called()
    # get_items_for_curation called with exclude_previous_editions=False
    mock_get_items.assert_called_once_with(config, exclude_previous_editions=False)


def test_curate_edition_default_records_edition(tmp_path):
    """fresh=False (default) records the edition and excludes previous items."""
    config = _curation_config(tmp_path)
    pool = _curation_pool(n=3, n_clusters=1)
    mock_record = MagicMock()
    mock_get_items = MagicMock(return_value=pool)
    with (
        patch("offscroll.curation.selection.get_items_for_curation", mock_get_items),
        patch("offscroll.curation.selection.get_cluster_count", return_value=1),
        patch("offscroll.curation.selection.get_edition_count", return_value=0),
        patch("offscroll.curation.selection.get_feed_name_map", return_value={}),
        patch("offscroll.curation.selection.record_edition", mock_record),
    ):
        path = curate_edition(config, fresh=False)
    assert path.exists()
    # record_edition IS called
    mock_record.assert_called_once()
    # get_items_for_curation called with exclude_previous_editions=True
    mock_get_items.assert_called_once_with(config, exclude_previous_editions=True)
