"""Loss function terms, page estimation, greedy optimizer, and curation pipeline.

Each loss term returns a float in [0.0, 1.0] where 0.0 is ideal.
The optimizer combines terms with configurable weights and selects
items via greedy initialization + swap hill-climbing.

curate_edition() pipeline entry point.
"""

from __future__ import annotations

import html as html_module
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import exp, log
from pathlib import Path

# Try to import embedding and clustering functions
try:
    from offscroll.ingestion.clustering import cluster_items
    from offscroll.ingestion.embeddings import embed_items

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    embed_items = None
    cluster_items = None

import re

from offscroll.curation.editorial import run_editorial
from offscroll.ingestion.store import (
    get_cluster_count,
    get_edition_count,
    get_feed_name_map,
    get_items_for_clustering,
    get_items_for_curation,
    get_items_for_embedding,
    record_edition,
    repair_missing_images,
    update_cluster_ids,
    update_embeddings,
)
from offscroll.models import (
    CuratedEdition,
    CuratedImage,
    CuratedItem,
    EditionMeta,
    FeedItem,
    LayoutHint,
    PullQuote,
    RankedEdition,
    RankedItem,
    Section,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "coverage": 1.0,
    "redundancy": 1.0,
    "quality": 1.0,
    "diversity": 1.0,
    "fit": 1.0,
}


@dataclass
class SelectionResult:
    """The output of the optimizer."""

    items: list[FeedItem]
    total_loss: float
    term_losses: dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    improved: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _quality_score(item: FeedItem) -> float:
    """Per-item quality score based on word count.

    Sigmoid centered at 20 words, rising to ~0.95 at 200 words.
    Clamped to [0.0, 1.0].
    """
    wc = item.word_count
    raw = 2.0 / (1.0 + exp(-0.02 * (wc - 20))) - 1.0
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Loss Function Terms
# ---------------------------------------------------------------------------


def coverage_loss(selection: list[FeedItem], n_clusters: int) -> float:
    """Fraction of clusters not represented in the selection.

    Args:
        selection: The selected items. Each must have cluster_id set.
        n_clusters: Total number of clusters (excluding noise cluster -1).

    Returns:
        0.0 if every cluster is represented, 1.0 if none are.
        Returns 0.0 if n_clusters == 0 or selection is empty.
    """
    if n_clusters == 0 or not selection:
        return 0.0
    represented = set()
    for item in selection:
        if item.cluster_id is not None and item.cluster_id != -1:
            represented.add(item.cluster_id)
    return 1.0 - len(represented) / n_clusters


def redundancy_loss(selection: list[FeedItem]) -> float:
    """Mean pairwise cosine similarity among items in the same cluster.

    For each pair of selected items that share a cluster_id (excluding
    noise cluster -1), compute cosine similarity of their embeddings.
    Return the mean. If no same-cluster pairs exist, return 0.0.

    Items with embedding == None are skipped.

    Args:
        selection: The selected items. Each must have cluster_id and
            embedding set.

    Returns:
        0.0 if no redundancy. Approaches 1.0 as items become identical.
    """
    # Group items by cluster, excluding noise and items without embeddings
    clusters: dict[int, list[FeedItem]] = {}
    for item in selection:
        if item.cluster_id is not None and item.cluster_id != -1 and item.embedding is not None:
            clusters.setdefault(item.cluster_id, []).append(item)

    similarities: list[float] = []
    for items in clusters.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sim = _cosine_similarity(items[i].embedding, items[j].embedding)  # type: ignore[arg-type]
                similarities.append(sim)

    if not similarities:
        return 0.0
    return sum(similarities) / len(similarities)


def quality_loss(selection: list[FeedItem]) -> float:
    """1.0 minus mean per-item quality score.

    Args:
        selection: The selected items.

    Returns:
        0.0 if all items have perfect quality. 1.0 if all items have
        zero quality. Returns 0.0 if selection is empty.
    """
    if not selection:
        return 0.0
    mean_quality = sum(_quality_score(item) for item in selection) / len(selection)
    return 1.0 - mean_quality


def diversity_loss(selection: list[FeedItem]) -> float:
    """1.0 minus normalized Shannon entropy over author distribution.

    Args:
        selection: The selected items. Items with author == None are
            treated as author "Unknown".

    Returns:
        0.0 if authors are perfectly uniformly distributed.
        1.0 if all items are from the same author.
        Returns 0.0 if selection has 0 or 1 items.
    """
    if len(selection) <= 1:
        return 0.0

    author_counts = Counter(
        item.author if item.author is not None else "Unknown" for item in selection
    )
    total = sum(author_counts.values())
    max_entropy = log(total)

    if max_entropy == 0.0:
        return 0.0

    entropy = 0.0
    for count in author_counts.values():
        p = count / total
        if p > 0.0:
            entropy -= p * log(p)

    return 1.0 - (entropy / max_entropy)


# ---------------------------------------------------------------------------
# Page Estimation
# ---------------------------------------------------------------------------


def estimate_pages(selection: list[FeedItem]) -> float:
    """Estimate how many pages the selection will fill.

    Rule of thumb: ~500 words per page in a 3-column newspaper layout
    at 10pt.

    Args:
        selection: The selected items.

    Returns:
        Estimated page count as a float.
    """
    total_words = sum(item.word_count for item in selection)
    return total_words / 500.0


def fit_loss(selection: list[FeedItem], target_pages: int = 10) -> float:
    """Absolute deviation from page target, normalized.

    Args:
        selection: The selected items.
        target_pages: Target page count (default 10).

    Returns:
        0.0 if estimated pages equals target. Increases linearly
        with deviation. Can exceed 1.0 if wildly over/under.
        Returns 0.0 if target_pages == 0 or selection is empty.
    """
    if target_pages == 0 or not selection:
        return 0.0
    estimated = estimate_pages(selection)
    return abs(estimated - target_pages) / target_pages


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------


def _combined_loss(
    selection: list[FeedItem],
    n_clusters: int,
    weights: dict[str, float],
    target_pages: int,
) -> tuple[float, dict[str, float]]:
    """Compute weighted combined loss and per-term breakdown."""
    losses = {
        "coverage": coverage_loss(selection, n_clusters),
        "redundancy": redundancy_loss(selection),
        "quality": quality_loss(selection),
        "diversity": diversity_loss(selection),
        "fit": fit_loss(selection, target_pages),
    }
    total = sum(weights[k] * losses[k] for k in losses)
    return total, losses


# ---------------------------------------------------------------------------
# Greedy Optimizer
# ---------------------------------------------------------------------------


def select_items(
    pool: list[FeedItem],
    n_clusters: int,
    weights: dict[str, float] | None = None,
    target_pages: int = 10,
    max_iterations: int = 500,
    min_items: int = 5,
) -> SelectionResult:
    """Select items from the pool to minimize combined loss.

    Algorithm:
        1. Greedy init: pick the highest-quality item from each cluster.
           Include noise items (cluster_id == -1) sorted by quality.
        2. Budget fit: if estimated pages > target, remove lowest-quality
           items (but never below min_items). If under, add
           highest-quality unselected items.
        3. Min-items guarantee: if fewer than min_items are selected
           after budget fitting, add highest-quality unselected items
           until min_items is reached, even if over budget.
        4. Hill-climb: for max_iterations, try swapping a selected item
           with an unselected item. Keep the swap if total loss decreases.

    Args:
        pool: All candidate FeedItem objects. Must have cluster_id and
            embedding set (items without these are skipped).
        n_clusters: Total number of non-noise clusters.
        weights: Loss term weights. Keys: "coverage", "redundancy",
            "quality", "diversity", "fit". Default: all 1.0.
        target_pages: Page budget target.
        max_iterations: Maximum swap iterations for hill-climbing.
        min_items: Minimum number of items to select.  
            : ensures content density even when a few long
            articles would fill the page budget alone.

    Returns:
        SelectionResult with the optimized selection.

    Raises:
        ValueError: If pool is empty.
    """
    if not pool:
        raise ValueError("Pool is empty")

    if weights is None:
        weights = _DEFAULT_WEIGHTS.copy()

    # Filter pool to usable items (have cluster_id set)
    usable = [item for item in pool if item.cluster_id is not None]
    if not usable:
        # All items lack cluster_id -- return best quality items
        usable = pool

    # --- Step 1: Greedy initialization ---
    # Pick the highest-quality item from each cluster
    selection: list[FeedItem] = []
    selected_ids: set[str] = set()

    # Group by cluster
    cluster_items: dict[int, list[FeedItem]] = {}
    noise_items: list[FeedItem] = []
    for item in usable:
        cid = item.cluster_id
        if cid is None or cid == -1:
            noise_items.append(item)
        else:
            cluster_items.setdefault(cid, []).append(item)

    # Pick best from each cluster
    for cid in range(n_clusters):
        candidates = cluster_items.get(cid, [])
        if candidates:
            best = max(candidates, key=_quality_score)
            selection.append(best)
            selected_ids.add(best.item_id)

    # Sort noise by quality descending for budget fitting
    noise_items.sort(key=_quality_score, reverse=True)

    # --- Step 2: Budget fitting ---
    over_budget = target_pages * 1.2
    under_budget = target_pages * 1.0

    # If over budget, remove lowest-quality items (but keep at least min_items)
    while estimate_pages(selection) > over_budget and len(selection) > min_items:
        worst_idx = min(range(len(selection)), key=lambda i: _quality_score(selection[i]))
        removed = selection.pop(worst_idx)
        selected_ids.discard(removed.item_id)

    # If under budget, add highest-quality unselected items
    unselected_pool = [item for item in usable if item.item_id not in selected_ids]
    unselected_pool.sort(key=_quality_score, reverse=True)

    for item in unselected_pool:
        if estimate_pages(selection) >= under_budget:
            break
        selection.append(item)
        selected_ids.add(item.item_id)

    # : Guarantee minimum article count for
    # content density.  If the page budget is satisfied with fewer
    # than min_items, keep adding highest-quality unselected items.
    if len(selection) < min_items:
        remaining = [item for item in usable if item.item_id not in selected_ids]
        remaining.sort(key=_quality_score, reverse=True)
        for item in remaining:
            if len(selection) >= min_items:
                break
            selection.append(item)
            selected_ids.add(item.item_id)

    # --- Step 3: Hill-climbing ---
    current_loss, current_terms = _combined_loss(selection, n_clusters, weights, target_pages)
    improved = False
    iterations = 0

    for _ in range(max_iterations):
        found_improvement = False
        unselected = [item for item in usable if item.item_id not in selected_ids]

        for sel_idx in range(len(selection)):
            for unsel_item in unselected:
                # Try swap
                trial = list(selection)
                old_item = trial[sel_idx]
                trial[sel_idx] = unsel_item

                trial_loss, trial_terms = _combined_loss(trial, n_clusters, weights, target_pages)
                if trial_loss < current_loss:
                    selection = trial
                    selected_ids.discard(old_item.item_id)
                    selected_ids.add(unsel_item.item_id)
                    current_loss = trial_loss
                    current_terms = trial_terms
                    improved = True
                    found_improvement = True
                    break
            if found_improvement:
                break

        iterations += 1
        if not found_improvement:
            break

    return SelectionResult(
        items=selection,
        total_loss=current_loss,
        term_losses=current_terms,
        iterations=iterations,
        improved=improved,
    )


# ---------------------------------------------------------------------------
# Layout Hint Assignment 
# ---------------------------------------------------------------------------


# : Pull quote quality rules.
# Never truncate mid-sentence, never select the first sentence,
# prefer strong declarative claims, min 8 words, max 40 words.
_STRONG_CLAIM_WORDS = frozenset([
    "is", "are", "was", "were", "must", "never", "always", "every",
    "nothing", "everything", "nobody", "turns out", "means", "proves",
    "shows", "reveals", "demands", "requires", "transforms",
])


#  Regex to fix subheading concatenation before pull
# quote selection. Matches "word.Uppercase" where a sentence-ending
# punctuation is immediately followed by an uppercase letter with
# no space. Same pattern as renderer._CONCAT_FIX_RE.
_CONCAT_FIX_RE = re.compile(r'([.!?])([A-Z])')


def _select_pull_quote(text: str) -> str | None:
    """Select the best pull quote sentence from article text.

    Quality rules for pull quotes:
    - Never truncate mid-sentence (only whole sentences)
    - Never select the first sentence
    - Prefer strong declarative claims
    - Min 8 words, max 40 words

     Fix concatenated sentences before splitting.
    Without this, "remarkable.This is" was parsed as one sentence,
    causing the DHH pull quote to be a multi-sentence blob and
    missing strong single-sentence candidates.

    Returns the best sentence or None.
    """
    if not text:
        return None

    # Fix concatenated sentences (e.g. "remarkable.This" -> "remarkable. This")
    text = _CONCAT_FIX_RE.sub(r'\1 \2', text)

    # Split into sentences on ". ", "! ", "? " boundaries
    # Keep the punctuation with each sentence
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(raw_sentences) < 2:
        return None

    # Skip the first sentence, filter by word count
    candidates = []
    for i, sent in enumerate(raw_sentences):
        if i == 0:
            continue  # Never select the first sentence
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        wc = len(words)
        if wc < 8 or wc > 40:
            continue
        # Must end with sentence-ending punctuation
        if sent[-1] not in '.!?':
            continue
        candidates.append(sent)

    if not candidates:
        return None

    # Score candidates: prefer sentences with strong claim words
    def _score(sent: str) -> float:
        lower = sent.lower()
        score = 0.0
        for word in _STRONG_CLAIM_WORDS:
            if word in lower:
                score += 1.0
        # Slight preference for medium-length sentences (15-25 words)
        wc = len(sent.split())
        if 15 <= wc <= 25:
            score += 0.5
        return score

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _assign_layout_hint(item: FeedItem, *, is_cover: bool = False) -> LayoutHint:
    """Assign a layout hint based on word count and cover status.

    Only the cover story gets FEATURE.
    All other articles get STANDARD or BRIEF, regardless of word
    count. This creates visual hierarchy -- one feature per edition.

    Args:
        item: The FeedItem to classify.
        is_cover: True if this item is the top-ranked (cover) story.

    Returns:
        FEATURE only for the cover story, BRIEF for 30 or fewer
        words, STANDARD otherwise.
    """
    if is_cover and item.word_count >= 300:
        return LayoutHint.FEATURE
    if item.word_count <= 30:
        return LayoutHint.BRIEF
    return LayoutHint.STANDARD


# ---------------------------------------------------------------------------
# Edition Builder 
# ---------------------------------------------------------------------------


def _convert_images(item: FeedItem) -> list[CuratedImage]:
    """Convert FeedItem images to CuratedImages.

    Only includes images that have a local_path (i.e., have been
    downloaded). Uses alt_text as caption, falling back to author
    attribution.
    """
    curated_images = []
    for img in item.images:
        if img.local_path is None:
            continue
        # : Use alt_text as caption if
        # available. If not, suppress the caption entirely rather
        # than generating a generic "Image from {author}" attribution
        # that tells the reader nothing.
        caption = img.alt_text or ""
        #  Unescape HTML entities in captions to prevent
        # double-escaping by Jinja2 autoescape
        caption = html_module.unescape(caption)
        curated_images.append(
            CuratedImage(
                local_path=img.local_path,
                caption=caption,
                width=img.width,
                height=img.height,
            )
        )
    return curated_images


def _build_curated_edition(
    result: SelectionResult,
    config: dict,
    pool_size: int = 0,
    feed_name_map: dict[str, str] | None = None,
) -> CuratedEdition:
    """Convert a SelectionResult into a CuratedEdition.

    Uses passthrough logic (no LLM):
    - display_text = content_text
    - title = existing title (may be None)
    - section headings from cluster_id ("Topic {n+1}")
    - layout_hint by word count heuristic
    - One pull quote from highest-quality item

    Args:
        result: The optimizer output.
        config: The OffScroll config dict.
        pool_size: Number of candidate items in the pool.
        feed_name_map: Optional mapping of feed_url -> feed_name.
            If None, will be loaded from the database.

    Returns:
        A complete CuratedEdition ready for rendering.
    """
    # Build feed URL -> name map for source attribution
    if feed_name_map is None:
        feed_name_map = get_feed_name_map(config)

    # Group items by cluster_id
    cluster_groups: dict[int, list[FeedItem]] = {}
    noise_items: list[FeedItem] = []

    for item in result.items:
        if item.cluster_id is None or item.cluster_id == -1:
            noise_items.append(item)
        else:
            cluster_groups.setdefault(item.cluster_id, []).append(item)

    # Build sections
    sections: list[Section] = []
    _cover_assigned = False  #  only one feature per edition

    # Cluster sections first
    for i, cid in enumerate(sorted(cluster_groups.keys())):
        curated_items = []
        for item in cluster_groups[cid]:
            if not item.content_text:
                continue
            _is_cover = not _cover_assigned and item.word_count >= 300
            if _is_cover:
                _cover_assigned = True
            curated_items.append(
                CuratedItem(
                    item_id=item.item_id,
                    display_text=html_module.unescape(item.content_text),
                    author=item.author or "Unknown",
                    author_url=item.author_url,
                    source_name=feed_name_map.get(item.feed_url),
                    item_url=item.item_url,
                    title=item.title,
                    images=_convert_images(item),
                    word_count=item.word_count,
                    layout_hint=_assign_layout_hint(item, is_cover=_is_cover),
                    cluster_id=item.cluster_id,
                    quality_score=_quality_score(item),
                )
            )
        if curated_items:
            sections.append(Section(heading=f"Topic {i + 1}", items=curated_items))

    # Noise section
    if noise_items:
        noise_curated = []
        for item in noise_items:
            if not item.content_text:
                continue
            _is_cover = not _cover_assigned and item.word_count >= 300
            if _is_cover:
                _cover_assigned = True
            noise_curated.append(
                CuratedItem(
                    item_id=item.item_id,
                    display_text=html_module.unescape(item.content_text),
                    author=item.author or "Unknown",
                    author_url=item.author_url,
                    source_name=feed_name_map.get(item.feed_url),
                    item_url=item.item_url,
                    title=item.title,
                    images=_convert_images(item),
                    word_count=item.word_count,
                    layout_hint=_assign_layout_hint(item, is_cover=_is_cover),
                    cluster_id=item.cluster_id,
                    quality_score=_quality_score(item),
                )
            )
        if noise_curated:
            sections.append(Section(heading="In Brief", items=noise_curated))

    # : Better pull quote selection
    pull_quotes: list[PullQuote] = []
    if result.items:
        for item in sorted(result.items, key=lambda it: it.word_count, reverse=True)[:5]:
            if not item.content_text:
                continue
            best = _select_pull_quote(item.content_text)
            if best:
                pull_quotes.append(
                    PullQuote(
                        text=best,
                        attribution=item.author or "Unknown",
                        source_item_id=item.item_id,
                    )
                )
            if len(pull_quotes) >= 3:
                break

    # Edition metadata
    newspaper_config = config.get("newspaper", {})
    title = newspaper_config.get("title", "The Morning Dispatch")
    subtitle_pattern = newspaper_config.get("subtitle_pattern", "Vol. {volume}, No. {issue}")
    issue_number = get_edition_count(config) + 1
    subtitle = subtitle_pattern.format(volume=1, issue=issue_number)
    page_target = newspaper_config.get("page_target", 10)

    edition_meta = EditionMeta(
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        title=title,
        subtitle=subtitle,
        editorial_note=None,
    )

    n_items = len(result.items)
    # : Reader-facing curation summary for
    # algorithmic transparency.  Format matches the Design POV spec:
    # "{n} articles selected from {pool} candidates across {topics} topics"
    n_sections = len(sections)
    curation_summary = (
        f"{n_items} articles selected from {pool_size} candidates"
        f" across {n_sections} topic{'s' if n_sections != 1 else ''}"
    )

    return CuratedEdition(
        edition=edition_meta,
        sections=sections,
        pull_quotes=pull_quotes,
        page_target=page_target,
        estimated_content_pages=estimate_pages(result.items),
        curation_summary=curation_summary,
    )


# ---------------------------------------------------------------------------
# Edition Validation ( task C2)
# ---------------------------------------------------------------------------

# LLM refusal patterns for headings and content
_VALIDATION_REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"i don'?t see",
        r"please provide",
        r"i cannot",
        r"i can'?t",
        r"sorry",
        r"i'?m not able",
        r"i'?m unable",
        r"as an ai",
        r"i don'?t have",
        r"no (?:content|text|information)",
    ]
]

# Boilerplate-only item patterns (short items that are just RSS remnants)
_BOILERPLATE_ITEM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^the post .+ first appeared on\b",
        r"^this essay .+ first appeared",
        r"^continue reading",
        r"^read more",
        r"^click to read",
    ]
]

# LLM preamble patterns for pull quotes
_PQ_PREAMBLE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^here (?:are|is)\b",
        r"^sure[!,.]",
        r"^the following\b",
        r"^certainly[!,.]",
        r"^of course[!,.]",
    ]
]


def validate_edition(edition: CuratedEdition) -> CuratedEdition:
    """Validate and clean a CuratedEdition before rendering.

    Runs between editorial processing and JSON serialization as a
    safety net. Removes:
    - Items with empty or zero-word display_text
    - Items under 20 words that match boilerplate patterns
    - Sections with headings that contain LLM refusal patterns
      (replaces with fallback heading)
    - Sections with headings over 80 characters (truncates)
    - Pull quotes that contain LLM preamble

    Args:
        edition: The CuratedEdition to validate.

    Returns:
        The cleaned edition (mutated in place and returned).
    """
    # Validate section headings
    for section in edition.sections:
        heading = section.heading.strip() if section.heading else ""
        # Check for LLM refusal in heading
        if any(pat.search(heading) for pat in _VALIDATION_REFUSAL_PATTERNS):
            # Fall back to first item's title or generic heading
            titles = [
                item.title
                for item in section.items
                if hasattr(item, "title") and item.title
            ]
            section.heading = titles[0][:40] if titles else "News"
            logger.warning("Replaced refusal heading with fallback: %s", section.heading)
        # Truncate overly long headings
        if len(section.heading) > 80:
            section.heading = section.heading[:77] + "..."

    # Validate items within sections
    for section in edition.sections:
        valid_items = []
        for item in section.items:
            if not hasattr(item, "display_text"):
                # Threads and other non-text items pass through
                valid_items.append(item)
                continue

            text = (item.display_text or "").strip()
            wc = getattr(item, "word_count", 0)

            # Reject zero-content items
            if not text or wc == 0:
                logger.info("Filtered zero-content item: %s", item.item_id)
                continue

            # Reject short boilerplate-only items
            if wc < 20 and any(
                pat.search(text) for pat in _BOILERPLATE_ITEM_PATTERNS
            ):
                logger.info("Filtered boilerplate item: %s", item.item_id)
                continue

            valid_items.append(item)
        section.items = valid_items

    # Remove empty sections
    edition.sections = [s for s in edition.sections if s.items]

    # Validate pull quotes
    valid_pqs = []
    for pq in edition.pull_quotes:
        text = (pq.text or "").strip()
        if not text:
            continue
        if any(pat.search(text) for pat in _PQ_PREAMBLE_PATTERNS):
            logger.info("Filtered preamble pull quote: %s", text[:50])
            continue
        valid_pqs.append(pq)
    edition.pull_quotes = valid_pqs

    return edition


# ---------------------------------------------------------------------------
# Ranked Edition Builder 
# ---------------------------------------------------------------------------


def rank_items(
    pool: list[FeedItem],
    n_clusters: int,
) -> list[tuple[FeedItem, float]]:
    """Rank ALL viable items by combined quality score.

    Unlike select_items(), this does not pick a fixed selection.
    It scores every item and returns them all, sorted by a combined
    score of quality, coverage importance, and diversity.

    The ranking formula:
    - Base score: quality_score (word-count sigmoid)
    - Coverage bonus: +0.15 if this item is the best in its cluster
    - Diversity penalty: -0.05 for each same-author item ranked above

    Args:
        pool: All candidate FeedItem objects.
        n_clusters: Total number of non-noise clusters.

    Returns:
        List of (FeedItem, score) tuples sorted by score descending.
    """
    if not pool:
        return []

    # Identify best item per cluster for coverage bonus
    cluster_best: dict[int, str] = {}
    cluster_items_map: dict[int, list[FeedItem]] = {}
    for item in pool:
        cid = item.cluster_id
        if cid is not None and cid != -1:
            cluster_items_map.setdefault(cid, []).append(item)

    for cid, items in cluster_items_map.items():
        best = max(items, key=_quality_score)
        cluster_best[cid] = best.item_id

    # Score each item
    scored: list[tuple[FeedItem, float]] = []
    for item in pool:
        score = _quality_score(item)

        # Coverage bonus: best-in-cluster gets a boost
        cid = item.cluster_id
        if cid is not None and cid != -1 and cluster_best.get(cid) == item.item_id:
            score += 0.15

        scored.append((item, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Apply diversity penalty (same-author items ranked lower)
    authors_seen: dict[str, int] = {}
    reranked: list[tuple[FeedItem, float]] = []
    for item, score in scored:
        author = item.author or "Unknown"
        count = authors_seen.get(author, 0)
        penalty = count * 0.05
        reranked.append((item, score - penalty))
        authors_seen[author] = count + 1

    # Re-sort after diversity penalty
    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked


def _assign_section_label(
    item: FeedItem,
    cluster_id: int | None,
    cluster_section_map: dict[int, str],
) -> str:
    """Assign a section label based on cluster_id.

    Cluster items get "Topic N" (to be refined by editorial).
    Noise items get "In Brief".
    """
    if cluster_id is not None and cluster_id != -1:
        return cluster_section_map.get(cluster_id, f"Topic {cluster_id + 1}")
    return "In Brief"


def _build_ranked_edition(
    pool: list[FeedItem],
    n_clusters: int,
    config: dict,
    feed_name_map: dict[str, str] | None = None,
) -> RankedEdition:
    """Build a RankedEdition from the full item pool.

    Ranks ALL viable items (not a fixed selection). The renderer
    will process items in rank order and stop when page_target
    is met.

    Args:
        pool: All candidate FeedItem objects (already filtered).
        n_clusters: Total number of non-noise clusters.
        config: The OffScroll config dict.
        feed_name_map: Optional mapping of feed_url -> feed_name.

    Returns:
        A RankedEdition with all items ranked.
    """
    if feed_name_map is None:
        feed_name_map = get_feed_name_map(config)

    # Rank all items
    ranked = rank_items(pool, n_clusters)

    # Build cluster -> section label map
    cluster_section_map: dict[int, str] = {}
    cluster_counter = 0
    for item, _ in ranked:
        cid = item.cluster_id
        if cid is not None and cid != -1 and cid not in cluster_section_map:
            cluster_counter += 1
            cluster_section_map[cid] = f"Topic {cluster_counter}"

    # Build ranked items
    ranked_items: list[RankedItem] = []
    for rank_pos, (item, score) in enumerate(ranked, start=1):
        # Skip empty items
        skip = False
        skip_reason = None
        if not item.content_text or not item.content_text.strip():
            skip = True
            skip_reason = "Empty content"
        elif item.word_count < 5:
            skip = True
            skip_reason = f"Too short ({item.word_count} words)"

        section = _assign_section_label(
            item, item.cluster_id, cluster_section_map
        )

        ranked_items.append(
            RankedItem(
                rank=rank_pos,
                item_id=item.item_id,
                layout_hint=_assign_layout_hint(item, is_cover=(rank_pos == 1)),
                section=section,
                display_text=html_module.unescape(item.content_text),
                title=item.title,
                author=item.author or "Unknown",
                author_url=item.author_url,
                source_name=feed_name_map.get(item.feed_url),
                item_url=item.item_url,
                images=_convert_images(item),
                word_count=item.word_count,
                cluster_id=item.cluster_id,
                quality_score=score,
                skip=skip,
                skip_reason=skip_reason,
            )
        )

    # Pull quote pool: best sentence from top 5 items
    # : Never truncate mid-sentence, never
    # select the first sentence, prefer strong declarative claims,
    # min 8 words, max 40 words.
    pull_quotes: list[PullQuote] = []
    for item, _ in ranked[:5]:
        if not item.content_text:
            continue
        best = _select_pull_quote(item.content_text)
        if best:
            pull_quotes.append(
                PullQuote(
                    text=best,
                    attribution=item.author or "Unknown",
                    source_item_id=item.item_id,
                )
            )
        if len(pull_quotes) >= 3:
            break

    # Edition metadata
    newspaper_config = config.get("newspaper", {})
    title = newspaper_config.get("title", "The Morning Dispatch")
    subtitle_pattern = newspaper_config.get(
        "subtitle_pattern", "Vol. {volume}, No. {issue}"
    )
    issue_number = get_edition_count(config) + 1
    subtitle = subtitle_pattern.format(volume=1, issue=issue_number)
    page_target = newspaper_config.get("page_target", 7)

    edition_meta = EditionMeta(
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        title=title,
        subtitle=subtitle,
        editorial_note=None,
    )

    n_viable = sum(1 for ri in ranked_items if not ri.skip)
    curation_summary = (
        f"{n_viable} viable items ranked from {len(pool)} candidates"
    )

    return RankedEdition(
        edition=edition_meta,
        ranked_items=ranked_items,
        pull_quote_pool=pull_quotes,
        page_target=page_target,
        curation_summary=curation_summary,
    )



def _apply_editorial_to_ranked(
    ranked: RankedEdition,
    editorial: CuratedEdition,
) -> None:
    """Apply editorial refinements from CuratedEdition back to RankedEdition.

    The editorial layer works on CuratedEdition (sections with items).
    After it runs, we need to propagate changes (section headings,
    editorial notes, layout hints, display_text) back to the ranked items.
    """
    # Build a lookup from item_id -> editorial CuratedItem
    editorial_items: dict[str, CuratedItem] = {}
    for section in editorial.sections:
        for item in section.items:
            if hasattr(item, "item_id"):
                editorial_items[item.item_id] = item

    # Build section heading map (old heading -> new heading)
    # The editorial layer may have renamed "Topic 1" to something better
    section_heading_map: dict[str, str] = {}
    for section in editorial.sections:
        # Match sections by their items' cluster_ids
        for item in section.items:
            if hasattr(item, "item_id") and item.item_id in editorial_items:
                # Find this item's original section in ranked
                for ri in ranked.ranked_items:
                    if ri.item_id == item.item_id:
                        if ri.section != section.heading:
                            section_heading_map[ri.section] = section.heading
                        break

    # Apply changes to ranked items
    for ri in ranked.ranked_items:
        # Update section headings
        if ri.section in section_heading_map:
            ri.section = section_heading_map[ri.section]

        # Apply editorial changes from matching CuratedItem
        ci = editorial_items.get(ri.item_id)
        if ci is not None:
            ri.editorial_note = ci.editorial_note
            ri.layout_hint = ci.layout_hint
            ri.selection_rationale = ci.selection_rationale
            ri.display_text = ci.display_text

    # Update edition-level editorial note
    ranked.edition.editorial_note = editorial.edition.editorial_note

    #  Keep heuristic pull quotes (which have quality
    # rules applied) rather than overriding with LLM-generated ones.
    # The LLM pull quotes from the editorial layer often select
    # truncated fragments or first sentences, which the heuristic
    # _select_pull_quote is specifically designed to avoid.
    # Only use editorial pull quotes if the heuristic produced none.
    if not ranked.pull_quote_pool and editorial.pull_quotes:
        ranked.pull_quote_pool = list(editorial.pull_quotes)


# ---------------------------------------------------------------------------
# Pipeline Entry Point ( ranked ordering)
# ---------------------------------------------------------------------------


def curate_edition(config: dict, fresh: bool = False) -> Path:
    """Run the full curation pipeline: embed -> cluster -> select -> editorial.

    Steps:
        1. Load un-embedded items from DB.
        2. Embed items (Ollama or stub).
        3. Persist embeddings to DB.
        4. Load un-clustered items from DB.
        5. Cluster items (HDBSCAN).
        6. Persist cluster_ids to DB.
        7. Load items ready for curation (embedded + clustered).
        8. Run optimizer (select_items).
        9. Build CuratedEdition.
        10. Run editorial layer (LLM polish).
        11. Write edition JSON.
        12. Record edition in DB.

    The embed and cluster steps are idempotent: items that already
    have embeddings/cluster_ids are skipped.

    Args:
        config: The OffScroll config dict.

    Returns:
        Path to the written edition JSON file.

    Raises:
        ValueError: If no items are available for curation.
    """
    # Step 1-3: Embedding pipeline
    if EMBEDDINGS_AVAILABLE and embed_items is not None:
        try:
            items_for_embedding = get_items_for_embedding(config)
            if items_for_embedding:
                logger.info(f"Embedding {len(items_for_embedding)} items")
                embedded = embed_items(items_for_embedding, config)
                update_embeddings(config, embedded)
                logger.info(f"Persisted embeddings for {len(embedded)} items")
        except Exception as e:
            logger.warning(f"Embedding failed, continuing with existing embeddings: {e}")
    else:
        logger.warning("Embedding functions not available, skipping embedding step")

    # Step 4-6: Clustering pipeline
    if EMBEDDINGS_AVAILABLE and cluster_items is not None:
        try:
            items_for_clustering = get_items_for_clustering(config)
            if items_for_clustering:
                logger.info(f"Clustering {len(items_for_clustering)} items")
                clustered = cluster_items(items_for_clustering, config)
                update_cluster_ids(config, clustered)
                logger.info(f"Persisted cluster_ids for {len(clustered)} items")
        except Exception as e:
            logger.warning(f"Clustering failed, continuing with existing cluster_ids: {e}")
    else:
        logger.warning("Clustering functions not available, skipping clustering step")

    # Step 6b: Repair items with missing images .
    # Items ingested before s HTML image extraction may
    # have empty images despite having <img> tags in content_html.
    try:
        repaired = repair_missing_images(config)
        if repaired:
            logger.info("Repaired images for %d items from content_html", repaired)
    except Exception as e:
        logger.warning("Image repair failed, continuing: %s", e)

    # Step 7: Load items ready for curation
    pool = get_items_for_curation(config, exclude_previous_editions=not fresh)
    n_clusters = get_cluster_count(config)

    if not pool:
        raise ValueError("No items available for curation")

    # Read config
    curation_config = config.get("curation", {})

    # Filter out low word count items
    # : Raised default from 10 to 200 words.
    # Single-paragraph RSS feed previews (120 words ending in "Source")
    # were admitted by the min_items=5 change, degrading the edition.
    # A newspaper article must have enough substance to justify its
    # presence as an independent piece. Items below the threshold do
    # not count toward min_items.
    min_word_count = curation_config.get("min_word_count", 200)
    pre_filter_count = len(pool)
    pool = [item for item in pool if item.word_count >= min_word_count]
    filtered_count = pre_filter_count - len(pool)
    if filtered_count > 0:
        logger.info("Filtered %d items below %d words", filtered_count, min_word_count)

    if not pool:
        raise ValueError("No items available for curation")
    weights = dict(curation_config.get("weights", _DEFAULT_WEIGHTS))
    max_iterations = curation_config.get("optimizer_iterations", 500)
    page_target = config.get("newspaper", {}).get("page_target", 10)
    # : Minimum article count for content density.
    min_items = curation_config.get("min_items", 5)

    # Step 8: Run optimizer
    result = select_items(
        pool,
        n_clusters,
        weights=weights,
        target_pages=page_target,
        max_iterations=max_iterations,
        min_items=min_items,
    )

    # Step 9: Build edition (load feed name map once for source attribution)
    feed_name_map = get_feed_name_map(config)
    edition = _build_curated_edition(
        result, config, pool_size=len(pool), feed_name_map=feed_name_map
    )

    # Step 10: Run editorial layer
    edition = run_editorial(edition, config)

    # Step 10b: Validate edition ( C2 safety net)
    edition = validate_edition(edition)

    # Step 10c: Defense-in-depth HTML entity decoding .
    # The editorial layer should not re-introduce entities, but if a
    # future LLM step modifies display_text, this catches it.
    for section in edition.sections:
        for item in section.items:
            if hasattr(item, "display_text") and item.display_text:
                item.display_text = html_module.unescape(item.display_text)

    # Step 11: Write JSON
    output_dir = Path(config["output"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"edition-{edition.edition.date}.json"
    edition.to_json(output_path)

    # Step 12: Record in DB (skip in fresh mode)
    if not fresh:
        item_ids = [item.item_id for item in result.items]
        edition_id = f"edition-{edition.edition.date}"
        record_edition(config, edition_id, item_ids, str(output_path))
    else:
        logger.info("Fresh mode: skipping edition recording")

    logger.info("Edition written to %s (%d items)", output_path, len(result.items))
    return output_path
