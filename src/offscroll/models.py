"""OffScroll shared data models.

This module defines the data contracts between components.
Changes to this file affect all components.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(Enum):
    """Where content was ingested from."""

    RSS = "rss"
    ATOM = "atom"
    MASTODON = "mastodon"
    BLUESKY = "bluesky"


class LayoutHint(Enum):
    """How the layout engine should render this item."""

    FEATURE = "feature"
    STANDARD = "standard"
    BRIEF = "brief"
    THREAD = "thread"


class CurationModel(Enum):
    """Which LLM backend to use for curation."""

    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"


class EmbeddingProvider(Enum):
    """Which embedding backend to use."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


# ---------------------------------------------------------------------------
# Ingested Content (Produced by ingestion, consumed by curation)
# ---------------------------------------------------------------------------


@dataclass
class ImageContent:
    """An image attached to a feed item."""

    url: str  # Original image URL
    local_path: str | None = None  # Path after download (relative to data dir)
    alt_text: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class FeedItem:
    """A single piece of content from any supported source.

    This is the source-agnostic content unit. It replaces
    CrawledTweet from the original architecture.
    """

    item_id: str  # Unique ID (feed-provided or generated)
    source_type: SourceType  # rss, atom, mastodon, bluesky
    feed_url: str  # The feed this came from
    item_url: str | None = None  # Permalink to original content
    author: str | None = None  # Author name or handle
    author_url: str | None = None  # Link to author profile
    title: str | None = None  # Item title (RSS/Atom have these; social posts may not)
    content_text: str = ""  # Plain text content
    content_html: str | None = None  # HTML content if available
    published_at: datetime | None = None  # When the item was published
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    images: list[ImageContent] = field(default_factory=list)
    is_thread: bool = False  # Part of a multi-item thread?
    thread_id: str | None = None  # Groups thread items together
    thread_position: int | None = None  # 1-indexed position in thread
    word_count: int = 0  # Computed on ingestion for layout budgeting
    embedding: list[float] | None = None  # Set after embedding step
    cluster_id: int | None = None  # Set after clustering step

    def __post_init__(self):
        if self.word_count == 0 and self.content_text:
            self.word_count = len(self.content_text.split())


@dataclass
class FeedSource:
    """A configured feed source."""

    url: str  # Feed URL
    source_type: SourceType
    name: str | None = None  # Human-readable name
    last_polled: datetime | None = None
    last_item_id: str | None = None  # For incremental polling


# ---------------------------------------------------------------------------
# Curated Edition (Produced by curation, consumed by layout)
# ---------------------------------------------------------------------------


@dataclass
class CuratedImage:
    """An image in the curated edition with caption."""

    local_path: str  # Relative to data dir
    caption: str  # LLM-generated or from alt text
    width: int | None = None
    height: int | None = None


@dataclass
class CuratedItem:
    """A single item selected for the edition."""

    item_id: str  # References the original FeedItem
    display_text: str  # Text to render (may be edited by LLM)
    author: str  # Display name or handle
    author_url: str | None = None
    source_name: str | None = None  # "Mastodon" or feed name
    item_url: str | None = None  # Permalink to original content
    title: str | None = None  # Headline (LLM may generate one)
    images: list[CuratedImage] = field(default_factory=list)
    editorial_note: str | None = None  # Context from the LLM
    layout_hint: LayoutHint = LayoutHint.STANDARD
    word_count: int = 0
    cluster_id: int | None = None
    quality_score: float | None = None
    selection_rationale: str | None = None


@dataclass
class CuratedThread:
    """A thread rendered as a single editorial unit."""

    thread_id: str
    headline: str  # LLM-generated thread headline
    author: str
    author_url: str | None = None
    items: list[CuratedItem] = field(default_factory=list)
    editorial_note: str | None = None
    layout_hint: LayoutHint = LayoutHint.THREAD


@dataclass
class PullQuote:
    """A striking sentence for visual emphasis."""

    text: str
    attribution: str
    source_item_id: str


@dataclass
class Section:
    """A thematic section of the edition."""

    heading: str
    items: list[CuratedItem | CuratedThread] = field(default_factory=list)


@dataclass
class EditionMeta:
    """Metadata for the edition."""

    date: str  # ISO date: "2026-03-01"
    title: str  # Newspaper name
    subtitle: str  # "Vol. 1, No. 12"
    editorial_note: str | None = None  # Opening editorial


@dataclass
class CuratedEdition:
    """The complete curated edition -- the JSON contract between
    curation and layout.

    This is the most
    important interface in the system.
    """

    edition: EditionMeta
    sections: list[Section] = field(default_factory=list)
    pull_quotes: list[PullQuote] = field(default_factory=list)
    page_target: int = 10
    estimated_content_pages: float = 0.0
    curation_summary: str | None = None

    def to_json(self, path: Path) -> None:
        """Serialize to JSON file."""

        def _default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Cannot serialize {type(obj)}")

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_default)

    @classmethod
    def from_json(cls, path: Path) -> CuratedEdition:
        """Deserialize from JSON file.

        This is how you load the curated edition in the
        renderer. The returned object has all the typed fields.
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses from dicts
        edition = EditionMeta(**data["edition"])

        sections = []
        for s in data.get("sections", []):
            items = []
            for item_data in s.get("items", []):
                # Determine if this is a thread or a single item
                if "thread_id" in item_data and "items" in item_data:
                    thread_items = [
                        CuratedItem(
                            **{
                                **ti,
                                "images": [CuratedImage(**img) for img in ti.get("images", [])],
                                "layout_hint": LayoutHint(ti.get("layout_hint", "standard")),
                            }
                        )
                        for ti in item_data["items"]
                    ]
                    items.append(
                        CuratedThread(
                            thread_id=item_data["thread_id"],
                            headline=item_data["headline"],
                            author=item_data["author"],
                            author_url=item_data.get("author_url"),
                            items=thread_items,
                            editorial_note=item_data.get("editorial_note"),
                            layout_hint=LayoutHint(item_data.get("layout_hint", "thread")),
                        )
                    )
                else:
                    items.append(
                        CuratedItem(
                            **{
                                **item_data,
                                "images": [
                                    CuratedImage(**img) for img in item_data.get("images", [])
                                ],
                                "layout_hint": LayoutHint(item_data.get("layout_hint", "standard")),
                            }
                        )
                    )
            sections.append(Section(heading=s["heading"], items=items))

        pull_quotes = [PullQuote(**pq) for pq in data.get("pull_quotes", [])]

        return cls(
            edition=edition,
            sections=sections,
            pull_quotes=pull_quotes,
            page_target=data.get("page_target", 10),
            estimated_content_pages=data.get("estimated_content_pages", 0.0),
            curation_summary=data.get("curation_summary"),
        )


# ---------------------------------------------------------------------------
# Ranked Edition ( new curation-renderer contract)
# ---------------------------------------------------------------------------


@dataclass
class RankedItem:
    """A single item in the ranked edition.

    The curation layer ranks all viable items by combined quality
    score. The renderer processes them in rank order and stops when
    the page target is met. This replaces the pre-selection model
    where curation chose a fixed number of items.
    """

    rank: int  # 1-based position in the ranked list
    item_id: str  # References the original FeedItem
    layout_hint: LayoutHint
    section: str  # Section heading this item belongs to
    display_text: str  # Text to render
    title: str | None = None
    author: str = "Unknown"
    author_url: str | None = None
    source_name: str | None = None
    item_url: str | None = None
    images: list[CuratedImage] = field(default_factory=list)
    editorial_note: str | None = None
    word_count: int = 0
    cluster_id: int | None = None
    quality_score: float | None = None
    selection_rationale: str | None = None
    skip: bool = False  # Flag to exclude from rendering
    skip_reason: str | None = None  # Why this item was skipped


@dataclass
class RankedEdition:
    """The ranked edition -- curation-renderer contract.

    Instead of pre-composed sections with a fixed selection, the
    curation layer produces a ranked list of ALL viable items. The
    renderer processes items in rank order, groups by section for
    headers, and stops when page_target is met.

    This eliminates dead-space problems because the renderer is the
    authority on what fits, not the curation layer's page estimator.
    """

    edition: EditionMeta
    ranked_items: list[RankedItem] = field(default_factory=list)
    pull_quote_pool: list[PullQuote] = field(default_factory=list)
    page_target: int = 7
    curation_summary: str | None = None

    def to_json(self, path: Path) -> None:
        """Serialize to JSON file."""

        def _default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Cannot serialize {type(obj)}")

        data = asdict(self)
        data["_format"] = "ranked"  # Format marker for detection
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_default)

    @classmethod
    def from_json(cls, path: Path) -> RankedEdition:
        """Deserialize from JSON file."""
        with open(path) as f:
            data = json.load(f)

        edition = EditionMeta(**data["edition"])

        ranked_items = []
        for item_data in data.get("ranked_items", []):
            ranked_items.append(
                RankedItem(
                    **{
                        **item_data,
                        "images": [
                            CuratedImage(**img)
                            for img in item_data.get("images", [])
                        ],
                        "layout_hint": LayoutHint(
                            item_data.get("layout_hint", "standard")
                        ),
                    }
                )
            )

        pull_quote_pool = [
            PullQuote(**pq) for pq in data.get("pull_quote_pool", [])
        ]

        return cls(
            edition=edition,
            ranked_items=ranked_items,
            pull_quote_pool=pull_quote_pool,
            page_target=data.get("page_target", 7),
            curation_summary=data.get("curation_summary"),
        )

    def to_curated_edition(self, placed_count: int | None = None) -> CuratedEdition:
        """Convert to a CuratedEdition for backward compatibility.

        Groups ranked items by section into Section objects. If
        placed_count is given, only include the first N non-skipped
        items (as the renderer would place them).

        This allows the existing renderer and editorial layer to
        work with ranked editions without modification.
        """
        items_to_place = [
            ri for ri in self.ranked_items if not ri.skip
        ]
        if placed_count is not None:
            items_to_place = items_to_place[:placed_count]

        # Group by section, preserving rank order
        section_map: dict[str, list[CuratedItem]] = {}
        section_order: list[str] = []
        for ri in items_to_place:
            if ri.section not in section_map:
                section_map[ri.section] = []
                section_order.append(ri.section)
            section_map[ri.section].append(
                CuratedItem(
                    item_id=ri.item_id,
                    display_text=ri.display_text,
                    author=ri.author,
                    author_url=ri.author_url,
                    source_name=ri.source_name,
                    item_url=ri.item_url,
                    title=ri.title,
                    images=list(ri.images),
                    editorial_note=ri.editorial_note,
                    layout_hint=ri.layout_hint,
                    word_count=ri.word_count,
                    cluster_id=ri.cluster_id,
                    quality_score=ri.quality_score,
                    selection_rationale=ri.selection_rationale,
                )
            )

        sections = [
            Section(heading=heading, items=section_map[heading])
            for heading in section_order
        ]

        return CuratedEdition(
            edition=self.edition,
            sections=sections,
            pull_quotes=list(self.pull_quote_pool),
            page_target=self.page_target,
            estimated_content_pages=0.0,
            curation_summary=self.curation_summary,
        )


def detect_edition_format(path: Path) -> str:
    """Detect whether a JSON file is a CuratedEdition or RankedEdition.

    Returns "ranked" if the file has a _format marker or ranked_items
    key, otherwise "curated".
    """
    with open(path) as f:
        data = json.load(f)
    if data.get("_format") == "ranked" or "ranked_items" in data:
        return "ranked"
    return "curated"


def load_edition(path: Path) -> CuratedEdition | RankedEdition:
    """Load an edition JSON file, auto-detecting the format.

    Returns either a CuratedEdition or RankedEdition based on the
    file contents.
    """
    fmt = detect_edition_format(path)
    if fmt == "ranked":
        return RankedEdition.from_json(path)
    return CuratedEdition.from_json(path)
