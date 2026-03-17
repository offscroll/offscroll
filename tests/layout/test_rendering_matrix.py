"""Rendering integration test matrix for OffScroll.

Task #90: Stress-test the rendering pipeline across diverse content
shapes to catch intermittent WeasyPrint and layout failures.

10 configurations varying:
- Feed types (RSS, Atom, Mastodon, Bluesky, mixed)
- Article counts (few: 5-10, moderate: 20-30, heavy: 50+)
- Content lengths (short tweets/toots, medium blog posts, long articles)

Each configuration is rendered at page targets 5, 10, and 20.
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path

import pytest

from offscroll.config import DEFAULTS
from offscroll.layout.renderer import _build_html, render_newspaper_pdf
from offscroll.models import (
    CuratedEdition,
    CuratedImage,
    CuratedItem,
    CuratedThread,
    EditionMeta,
    LayoutHint,
    PullQuote,
    Section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content generators
# ---------------------------------------------------------------------------

# Realistic sentence fragments for content generation
_SENTENCES_SHORT = [
    "Breaking: major policy shift announced today.",
    "The community responded with enthusiasm.",
    "Worth reading if you care about open standards.",
    "This changes everything we thought we knew.",
    "A small but significant step forward.",
    "Interesting thread on decentralization.",
    "Hot take: this is the future of the web.",
    "Just published my thoughts on this topic.",
    "The data speaks for itself here.",
    "Cannot overstate how important this is.",
]

_SENTENCES_MEDIUM = [
    (
        "The open web continues to evolve in unexpected directions, with new protocols "
        "emerging that challenge the dominance of centralized platforms. Researchers at "
        "several universities have published findings suggesting that federated systems "
        "can achieve comparable performance to their centralized counterparts."
    ),
    (
        "In a surprising turn of events, the regulatory landscape shifted dramatically "
        "this week as multiple jurisdictions announced coordinated approaches to digital "
        "sovereignty. The implications for technology companies operating across borders "
        "are significant and far-reaching."
    ),
    (
        "Community-driven open source projects have seen a remarkable surge in contributions "
        "over the past quarter, with several key infrastructure libraries receiving major "
        "updates. The trend suggests a growing recognition that shared infrastructure "
        "benefits everyone in the ecosystem."
    ),
    (
        "The intersection of privacy technology and user experience continues to challenge "
        "designers and engineers alike. Recent usability studies show that the most "
        "privacy-preserving options are often the least intuitive, creating a tension "
        "that the industry must resolve."
    ),
    (
        "Agricultural innovation is increasingly drawing on traditional practices, with "
        "regenerative farming methods demonstrating yields competitive with conventional "
        "approaches while building soil health over time. The economic case for transition "
        "grows stronger with each published study."
    ),
]

_SENTENCES_LONG = [
    (
        "The question of individual sovereignty in the digital age extends far beyond "
        "data ownership, touching on fundamental aspects of identity, agency, and the "
        "relationship between individuals and the institutions that shape their lives. "
        "As we build new systems and protocols, we must ask not just whether they work "
        "technically, but whether they serve the deeper goal of enabling people to live "
        "according to their own values and judgments. This is not a purely technical "
        "question; it requires us to think carefully about power, access, and the "
        "distribution of capability in an increasingly networked world. The tools we "
        "build today will shape the options available to future generations, and we "
        "bear a responsibility to ensure those options include genuine autonomy."
    ),
    (
        "Modern publishing workflows have undergone a dramatic transformation over "
        "the past decade, but the fundamental challenge remains the same: how do you "
        "take raw information and present it in a way that respects the reader's time "
        "and attention? The newspaper metaphor endures because it solved this problem "
        "elegantly -- curation, hierarchy, and visual design work together to guide "
        "the reader through a complex information landscape. Digital tools can enhance "
        "this process but cannot replace the editorial judgment that makes it work. "
        "The best algorithms in the world cannot substitute for a thoughtful editor "
        "who understands their audience and the material they are presenting."
    ),
    (
        "Self-hosting has moved from a niche hobby to a practical option for many "
        "people, driven by improvements in hardware, software, and documentation. "
        "The Raspberry Pi generation grew up tinkering, and they are now building "
        "the infrastructure that lets ordinary people run their own services without "
        "relying on corporate platforms. This is not about replacing every cloud "
        "service -- it is about having the option, the capability, and the knowledge "
        "to run your own when it matters. Email, file storage, DNS, web hosting: "
        "these are the building blocks of digital independence, and they are more "
        "accessible than ever before. The tooling has matured, the communities are "
        "welcoming, and the cost has dropped to the point where the main barrier "
        "is no longer technical but motivational."
    ),
]


def _make_short_text(word_target: int = 25) -> str:
    """Generate short content (tweet/toot length, 15-40 words)."""
    return random.choice(_SENTENCES_SHORT)


def _make_medium_text(word_target: int = 200) -> str:
    """Generate medium blog post content (150-400 words)."""
    paragraphs = []
    words = 0
    while words < word_target:
        sent = random.choice(_SENTENCES_MEDIUM)
        paragraphs.append(sent)
        words += len(sent.split())
    return "\n\n".join(paragraphs)


def _make_long_text(word_target: int = 800) -> str:
    """Generate long article content (500-2000 words)."""
    paragraphs = []
    words = 0
    while words < word_target:
        sent = random.choice(_SENTENCES_LONG)
        paragraphs.append(sent)
        words += len(sent.split())
    return "\n\n".join(paragraphs)


def _make_item(
    idx: int,
    source: str = "rss",
    length: str = "medium",
    layout: LayoutHint = LayoutHint.STANDARD,
    with_image: bool = False,
    with_title: bool = True,
) -> CuratedItem:
    """Build a single CuratedItem with realistic content."""
    authors = {
        "rss": [
            "Alice Chen",
            "Bob Martinez",
            "Carol Williams",
            "David Kim",
            "Eve Johnson",
        ],
        "atom": [
            "Frank Lee",
            "Grace Patel",
            "Hector Ramirez",
            "Irene Sokolov",
            "Jack Thompson",
        ],
        "mastodon": [
            "@ada@hachyderm.io",
            "@soren@fosstodon.org",
            "@maya@mastodon.social",
            "@kai@scholar.social",
            "@gus@indieweb.social",
        ],
        "bluesky": [
            "ralph.bsky.social",
            "jim.bsky.social",
            "elena.bsky.social",
            "hazel.bsky.social",
            "neville.bsky.social",
        ],
    }

    text_generators = {
        "short": _make_short_text,
        "medium": _make_medium_text,
        "long": _make_long_text,
    }

    text = text_generators[length]()
    author = random.choice(authors.get(source, authors["rss"]))

    title = None
    if with_title:
        titles = [
            "The Future of Open Standards",
            "Why Decentralization Matters",
            "Notes on Digital Sovereignty",
            "Regenerative Approaches to Technology",
            "Building for Independence",
            "The Case for Self-Hosting",
            "Community-Driven Infrastructure",
            "Lessons from the Fediverse",
            "Privacy by Design",
            "The Editorial Challenge",
        ]
        title = random.choice(titles) + f" #{idx}"

    images = []
    if with_image:
        images = [
            CuratedImage(
                local_path="images/placeholder.jpg",
                caption=f"Illustration for article {idx}",
                width=800,
                height=600,
            )
        ]

    source_names = {
        "rss": "Tech Blog",
        "atom": "Research Journal",
        "mastodon": "Mastodon",
        "bluesky": "Bluesky",
    }

    return CuratedItem(
        item_id=f"{source}-{idx:04d}",
        display_text=text,
        author=author,
        author_url=f"https://example.com/author/{idx}",
        source_name=source_names.get(source, "Web"),
        item_url=f"https://example.com/post/{idx}",
        title=title,
        images=images,
        editorial_note="Selected for its coverage of emerging trends." if idx % 5 == 0 else None,
        layout_hint=layout,
        word_count=len(text.split()),
    )


def _make_thread(idx: int, source: str = "mastodon", post_count: int = 5) -> CuratedThread:
    """Build a CuratedThread with multiple posts."""
    items = []
    for i in range(post_count):
        items.append(
            CuratedItem(
                item_id=f"{source}-thread-{idx}-{i}",
                display_text=_make_short_text(),
                author=f"@threader@{source}.social",
                layout_hint=LayoutHint.BRIEF,
                word_count=25,
            )
        )
    return CuratedThread(
        thread_id=f"thread-{idx}",
        headline=f"Thread: Exploring Topic #{idx}",
        author=f"@threader@{source}.social",
        author_url=f"https://{source}.social/@threader",
        items=items,
        editorial_note="A compelling thread worth reading in full.",
        layout_hint=LayoutHint.THREAD,
    )


def _make_pull_quote(item_id: str) -> PullQuote:
    """Generate a pull quote referencing an item."""
    quotes = [
        "The best test is the one that catches the bug you didn't expect.",
        "Individual sovereignty is not a feature -- it is the foundation.",
        "We build tools so that people can build their own futures.",
        "The open web is not dead; it is waiting to be rebuilt.",
        "Curation is an act of care, not just filtering.",
    ]
    return PullQuote(
        text=random.choice(quotes),
        attribution="A. Author",
        source_item_id=item_id,
    )


def _build_edition(
    name: str,
    sources: list[str],
    item_count: int,
    length_dist: dict[str, float],
    include_threads: bool = False,
    include_images: bool = True,
    thread_count: int = 0,
) -> CuratedEdition:
    """Build a complete CuratedEdition for testing.

    Args:
        name: Edition name for identification.
        sources: List of source types to draw from.
        item_count: Total number of items to generate.
        length_dist: Distribution of content lengths,
                     e.g. {"short": 0.5, "medium": 0.3, "long": 0.2}
        include_threads: Whether to include thread items.
        include_images: Whether some items get images.
        thread_count: Number of threads to add.
    """
    random.seed(hash(name) % 2**31)  # Reproducible per config

    items: list[CuratedItem | CuratedThread] = []

    # Generate regular items
    for i in range(item_count):
        source = random.choice(sources)
        # Pick length based on distribution
        r = random.random()
        cumulative = 0.0
        length = "medium"
        for len_key, prob in length_dist.items():
            cumulative += prob
            if r <= cumulative:
                length = len_key
                break

        # First item in each edition is a feature
        if i == 0:
            layout = LayoutHint.FEATURE
            length = "long"
        elif length == "short":
            layout = LayoutHint.BRIEF
        else:
            layout = LayoutHint.STANDARD

        has_title = source in ("rss", "atom") or random.random() > 0.5
        has_image = include_images and random.random() > 0.6

        items.append(
            _make_item(
                idx=i,
                source=source,
                length=length,
                layout=layout,
                with_image=has_image,
                with_title=has_title,
            )
        )

    # Add threads
    threads: list[CuratedThread] = []
    for t in range(thread_count):
        source = random.choice([s for s in sources if s in ("mastodon", "bluesky")] or sources)
        threads.append(_make_thread(idx=t, source=source))

    # Group items into sections
    section_names = [
        "Top Stories",
        "Technology",
        "Culture & Society",
        "The Fediverse",
        "In Brief",
    ]

    sections: list[Section] = []
    items_per_section = max(1, len(items) // min(len(section_names), max(2, len(items) // 3)))

    for i, sec_name in enumerate(section_names):
        start = i * items_per_section
        end = start + items_per_section
        sec_items: list[CuratedItem | CuratedThread] = list(items[start:end])

        # Add threads to relevant sections
        if sec_name == "The Fediverse" and threads:
            sec_items.extend(threads)
        elif sec_name == "In Brief" and not threads:
            # Convert remaining items to briefs
            for item in sec_items:
                if isinstance(item, CuratedItem):
                    item.layout_hint = LayoutHint.BRIEF

        if sec_items:
            sections.append(Section(heading=sec_name, items=sec_items))

    # Append any remaining items not assigned to sections
    remaining = items[len(section_names) * items_per_section :]
    if remaining:
        sections.append(Section(heading="More Stories", items=list(remaining)))

    # Generate pull quotes from feature/standard items
    pull_quotes = []
    quote_candidates = [
        it for it in items if isinstance(it, CuratedItem) and it.layout_hint != LayoutHint.BRIEF
    ]
    for item in quote_candidates[:3]:
        pull_quotes.append(_make_pull_quote(item.item_id))

    return CuratedEdition(
        edition=EditionMeta(
            date="2026-03-17",
            title=f"Test Gazette: {name}",
            subtitle="Integration Test Edition",
            editorial_note=f"Test configuration: {name}",
        ),
        sections=sections,
        pull_quotes=pull_quotes,
        page_target=10,
    )


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

# Each tuple: (name, sources, item_count, length_dist, threads, images, thread_count)
CONFIGURATIONS = [
    # 1. RSS-only, few items, medium length
    (
        "rss_few_medium",
        ["rss"],
        7,
        {"short": 0.0, "medium": 1.0, "long": 0.0},
        False,
        True,
        0,
    ),
    # 2. Atom-only, moderate items, mixed lengths
    (
        "atom_moderate_mixed",
        ["atom"],
        25,
        {"short": 0.2, "medium": 0.5, "long": 0.3},
        False,
        True,
        0,
    ),
    # 3. Mastodon-only, many short toots
    (
        "mastodon_many_short",
        ["mastodon"],
        55,
        {"short": 0.8, "medium": 0.2, "long": 0.0},
        True,
        False,
        3,
    ),
    # 4. Bluesky-only, moderate posts, short-medium
    (
        "bluesky_moderate_short",
        ["bluesky"],
        20,
        {"short": 0.6, "medium": 0.4, "long": 0.0},
        False,
        False,
        0,
    ),
    # 5. Mixed RSS+Atom, few items, all long articles
    (
        "mixed_rss_atom_few_long",
        ["rss", "atom"],
        8,
        {"short": 0.0, "medium": 0.0, "long": 1.0},
        False,
        True,
        0,
    ),
    # 6. Mixed all sources, moderate items, even distribution
    (
        "mixed_all_moderate_even",
        ["rss", "atom", "mastodon", "bluesky"],
        30,
        {"short": 0.33, "medium": 0.34, "long": 0.33},
        True,
        True,
        2,
    ),
    # 7. Heavy RSS, many long articles (stress test)
    (
        "rss_heavy_long",
        ["rss"],
        50,
        {"short": 0.0, "medium": 0.3, "long": 0.7},
        False,
        True,
        0,
    ),
    # 8. Mixed social, heavy short content with threads
    (
        "social_heavy_threads",
        ["mastodon", "bluesky"],
        60,
        {"short": 0.7, "medium": 0.3, "long": 0.0},
        True,
        False,
        5,
    ),
    # 9. Minimal: 5 items, single source, no images
    (
        "minimal_no_images",
        ["rss"],
        5,
        {"short": 0.0, "medium": 0.5, "long": 0.5},
        False,
        False,
        0,
    ),
    # 10. Maximum diversity: all sources, heavy count, all lengths, threads, images
    (
        "max_diversity",
        ["rss", "atom", "mastodon", "bluesky"],
        65,
        {"short": 0.3, "medium": 0.4, "long": 0.3},
        True,
        True,
        4,
    ),
]

PAGE_TARGETS = [5, 10, 20]


def _make_config(tmp_path: Path, page_target: int) -> dict:
    """Build a test config dict."""
    config = copy.deepcopy(DEFAULTS)
    config["newspaper"]["page_target"] = page_target
    config["output"]["data_dir"] = str(tmp_path)
    config["logging"]["level"] = "WARNING"
    config["logging"]["file"] = None
    return config


# ---------------------------------------------------------------------------
# Parametrized integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=CONFIGURATIONS, ids=[c[0] for c in CONFIGURATIONS])
def edition_config(request):
    """Fixture that yields (config_name, CuratedEdition) for each configuration."""
    name, sources, count, length_dist, threads, images, thread_count = request.param
    edition = _build_edition(
        name=name,
        sources=sources,
        item_count=count,
        length_dist=length_dist,
        include_threads=threads,
        include_images=images,
        thread_count=thread_count,
    )
    return name, edition


class TestRenderingMatrixHTML:
    """Test HTML rendering across all configurations and page targets."""

    @pytest.mark.parametrize("page_target", PAGE_TARGETS, ids=[f"pages_{p}" for p in PAGE_TARGETS])
    def test_html_renders_without_error(self, edition_config, page_target, tmp_path):
        """Each configuration produces valid HTML at each page target."""
        name, edition = edition_config
        edition.page_target = page_target
        config = _make_config(tmp_path, page_target)

        html = _build_html(edition, config)

        assert isinstance(html, str), f"[{name}] _build_html returned non-string"
        assert len(html) > 100, f"[{name}] HTML output too short ({len(html)} chars)"
        assert "<html" in html, f"[{name}] Missing <html> tag"
        assert "</html>" in html, f"[{name}] Missing </html> tag"

    @pytest.mark.parametrize("page_target", PAGE_TARGETS, ids=[f"pages_{p}" for p in PAGE_TARGETS])
    def test_html_contains_edition_title(self, edition_config, page_target, tmp_path):
        """The edition title appears in the rendered HTML."""
        name, edition = edition_config
        edition.page_target = page_target
        config = _make_config(tmp_path, page_target)

        html = _build_html(edition, config)
        assert edition.edition.title in html, f"[{name}] Title missing from HTML"

    @pytest.mark.parametrize("page_target", PAGE_TARGETS, ids=[f"pages_{p}" for p in PAGE_TARGETS])
    def test_html_contains_sections(self, edition_config, page_target, tmp_path):
        """Each section heading appears in the rendered HTML."""
        import html as html_module

        name, edition = edition_config
        edition.page_target = page_target
        config = _make_config(tmp_path, page_target)

        html = _build_html(edition, config)
        for section in edition.sections:
            # Jinja2 autoescape converts & -> &amp; etc.
            escaped_heading = html_module.escape(section.heading)
            assert escaped_heading in html, (
                f"[{name}] Section '{section.heading}' missing from HTML"
            )

    @pytest.mark.parametrize("page_target", PAGE_TARGETS, ids=[f"pages_{p}" for p in PAGE_TARGETS])
    def test_html_contains_css(self, edition_config, page_target, tmp_path):
        """Rendered HTML contains inlined CSS styles."""
        name, edition = edition_config
        edition.page_target = page_target
        config = _make_config(tmp_path, page_target)

        html = _build_html(edition, config)
        assert "<style" in html, f"[{name}] No <style> block in HTML"


class TestRenderingMatrixPDF:
    """Test PDF rendering across all configurations and page targets.

    These tests exercise WeasyPrint and catch rendering failures
    that only surface with specific content shapes.
    """

    @pytest.mark.parametrize("page_target", PAGE_TARGETS, ids=[f"pages_{p}" for p in PAGE_TARGETS])
    def test_pdf_renders_without_error(self, edition_config, page_target, tmp_path):
        """Each configuration produces a valid PDF at each page target."""
        name, edition = edition_config
        edition.page_target = page_target
        config = _make_config(tmp_path, page_target)

        pdf_path = render_newspaper_pdf(config, edition=edition)

        assert pdf_path.exists(), f"[{name}] PDF file not created"
        assert pdf_path.stat().st_size > 1000, (
            f"[{name}] PDF too small ({pdf_path.stat().st_size} bytes)"
        )

    @pytest.mark.parametrize("page_target", PAGE_TARGETS, ids=[f"pages_{p}" for p in PAGE_TARGETS])
    def test_pdf_is_valid(self, edition_config, page_target, tmp_path):
        """PDF starts with the correct magic bytes."""
        name, edition = edition_config
        edition.page_target = page_target
        config = _make_config(tmp_path, page_target)

        pdf_path = render_newspaper_pdf(config, edition=edition)

        with open(pdf_path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-", f"[{name}] PDF has invalid header: {header!r}"


class TestEdgeCases:
    """Test specific edge cases that stress the rendering pipeline."""

    def test_empty_sections(self, tmp_path):
        """Edition with empty sections does not crash."""
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="Empty Section Test",
                subtitle="Edge Case",
            ),
            sections=[
                Section(
                    heading="Has Items",
                    items=[
                        _make_item(0, length="medium"),
                    ],
                ),
                Section(heading="Empty Section", items=[]),
                Section(
                    heading="Also Has Items",
                    items=[
                        _make_item(1, length="short", layout=LayoutHint.BRIEF),
                    ],
                ),
            ],
            page_target=5,
        )
        config = _make_config(tmp_path, 5)
        html = _build_html(edition, config)
        assert "<html" in html

    def test_no_feature_article(self, tmp_path):
        """Edition with no FEATURE layout hint renders successfully."""
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="No Feature Test",
                subtitle="Edge Case",
            ),
            sections=[
                Section(
                    heading="Stories",
                    items=[
                        _make_item(i, length="medium", layout=LayoutHint.STANDARD) for i in range(5)
                    ],
                ),
            ],
            page_target=5,
        )
        config = _make_config(tmp_path, 5)
        html = _build_html(edition, config)
        assert "<html" in html

    def test_all_briefs(self, tmp_path):
        """Edition with only brief items renders successfully."""
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="All Briefs Test",
                subtitle="Edge Case",
            ),
            sections=[
                Section(
                    heading="In Brief",
                    items=[
                        _make_item(i, length="short", layout=LayoutHint.BRIEF, with_title=False)
                        for i in range(15)
                    ],
                ),
            ],
            page_target=5,
        )
        config = _make_config(tmp_path, 5)
        html = _build_html(edition, config)
        assert "<html" in html

    def test_single_very_long_article(self, tmp_path):
        """Single article with 3000+ words renders without overflow issues."""
        long_text = _make_long_text(word_target=3000)
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="Long Article Test",
                subtitle="Edge Case",
            ),
            sections=[
                Section(
                    heading="Feature",
                    items=[
                        CuratedItem(
                            item_id="long-001",
                            display_text=long_text,
                            author="Prolific Writer",
                            title="An Extremely Long Feature Article",
                            layout_hint=LayoutHint.FEATURE,
                            word_count=len(long_text.split()),
                        ),
                    ],
                ),
            ],
            page_target=5,
        )
        config = _make_config(tmp_path, 5)
        pdf_path = render_newspaper_pdf(config, edition=edition)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000

    def test_many_images(self, tmp_path):
        """Items with multiple images do not break layout."""
        items = []
        for i in range(5):
            item = _make_item(i, length="medium", with_image=False)
            item.images = [
                CuratedImage(
                    local_path="images/placeholder.jpg",
                    caption=f"Image {j} for article {i}",
                    width=800,
                    height=600,
                )
                for j in range(4)
            ]
            items.append(item)

        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="Multi-Image Test",
                subtitle="Edge Case",
            ),
            sections=[Section(heading="Gallery", items=items)],
            page_target=10,
        )
        config = _make_config(tmp_path, 10)
        html = _build_html(edition, config)
        assert "<html" in html

    def test_threads_only(self, tmp_path):
        """Edition with only thread content renders successfully."""
        threads = [_make_thread(i, post_count=8) for i in range(3)]
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="Threads Only Test",
                subtitle="Edge Case",
            ),
            sections=[
                Section(heading="Conversations", items=threads),
            ],
            page_target=5,
        )
        config = _make_config(tmp_path, 5)
        html = _build_html(edition, config)
        assert "<html" in html

    def test_unicode_content(self, tmp_path):
        """Content with Unicode characters renders without encoding errors."""
        unicode_item = CuratedItem(
            item_id="unicode-001",
            display_text=(
                "Caf\u00e9 culture meets \u6280\u8853 innovation. "
                "The \u00fcber-connected world needs \u2014 and deserves \u2014 "
                "tools built with \u2764\ufe0f for humanity. "
                "\u00bfPor qu\u00e9 no los dos? \u00c9galitaire et libre."
            ),
            author="\u00c9milie Dupont-\u5c71\u7530",
            title="Unicode \u2014 A Global Test \u2192 Success",
            layout_hint=LayoutHint.STANDARD,
            word_count=30,
        )
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="Unicode \u2014 Test Gazette",
                subtitle="Vol. 1, \u2116 1",
            ),
            sections=[Section(heading="\u00c0 la Une", items=[unicode_item])],
            page_target=5,
        )
        config = _make_config(tmp_path, 5)
        pdf_path = render_newspaper_pdf(config, edition=edition)
        assert pdf_path.exists()

    def test_special_html_characters(self, tmp_path):
        """Content with HTML special characters does not break rendering."""
        html_item = CuratedItem(
            item_id="html-001",
            display_text=(
                'The <script>alert("xss")</script> problem is real. '
                "Use &amp; instead of & and don't forget <img> tags. "
                "Also: 1 < 2 and 3 > 2 are both true. "
                "Quotes: \"double\" and 'single' work fine."
            ),
            author="Security Researcher",
            title='Testing <b>HTML</b> & "Escaping"',
            layout_hint=LayoutHint.STANDARD,
            word_count=40,
        )
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="HTML Escaping Test",
                subtitle="Edge Case",
            ),
            sections=[Section(heading="Security", items=[html_item])],
            page_target=5,
        )
        config = _make_config(tmp_path, 5)
        html = _build_html(edition, config)
        assert "<html" in html
        # Script tags should be escaped, not executable
        assert "<script>" not in html

    def test_mixed_layout_hints_in_section(self, tmp_path):
        """A section with mixed FEATURE, STANDARD, BRIEF, THREAD items."""
        items: list[CuratedItem | CuratedThread] = [
            _make_item(0, length="long", layout=LayoutHint.FEATURE),
            _make_item(1, length="medium", layout=LayoutHint.STANDARD),
            _make_item(2, length="medium", layout=LayoutHint.STANDARD),
            _make_item(3, length="short", layout=LayoutHint.BRIEF),
            _make_item(4, length="short", layout=LayoutHint.BRIEF),
            _make_thread(0, post_count=4),
        ]
        edition = CuratedEdition(
            edition=EditionMeta(
                date="2026-03-17",
                title="Mixed Layout Test",
                subtitle="Edge Case",
            ),
            sections=[Section(heading="Mixed Content", items=items)],
            page_target=10,
        )
        config = _make_config(tmp_path, 10)
        pdf_path = render_newspaper_pdf(config, edition=edition)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000
