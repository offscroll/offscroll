# Typographic Diversity: Design Vision for OffScroll

**Author:** Neville (Layout and Publishing Expert)
**Task:** #392 — Dimensions of variation for the Typst template
system
**Status:** Design specification, awaiting architect implementation
**Audience:** Ada (architect), Priya (frontend), Erik (founder)

---

## 0. The Problem, Stated Plainly

OffScroll currently has one typographic system. Six font sizes, two
families, two weights, one set of rules and gutters, one masthead
treatment, one drop cap. Every issue prints the same chord, and the
only thing that varies is which notes get played in what order. The
ML grader cannot learn what good hierarchy looks like because the
corpus is monotonal — and more importantly, the *newspaper itself*
is denied the editorial range that makes a printed page worth
holding.

The fix is not to add knobs and twiddle them randomly. Random
variation produces noise, not design. The fix is to define a small
set of **coherent typographic systems** and a small set of
**responsive variations within each system**, both governed by
editorial principles a thoughtful designer would actually defend.

This document specifies those systems and variations. The unit is
not "a single value of a parameter" but "a bundle of mutually
consistent choices." That is what an architect needs to
parameterize, and that is the only kind of variation a model can
learn from.

---

## 1. Design Philosophy: Structured Variation, Not Random Variation

Three commitments precede every dimension below.

### 1.1. Variation Travels in Bundles

A page where headlines are a tight 1.4× scale ratio, with hairline
section rules and 0.06in inter-paragraph spacing, looks coherent.
The same page with a 2.8× scale ratio, double-rule section breaks,
and 0.16in inter-paragraph spacing also looks coherent. Mix them
and you get a broken page.

Therefore: parameters do not vary independently. They cluster. The
architect's job is not to expose 47 individual parameters but to
expose a small number of **typographic systems** (coherent bundles)
and a small number of **moods** (coherent within-system
variations).

### 1.2. The Cadence of Variation Matters

Some things should vary per edition (the typographic personality of
the issue). Some should vary per section (the visual register of
"Features" vs. "In Brief"). Some should vary per article (the
treatment that separates the lead from the rest of the page). And
some should not vary at all (body type at 9.5–10.5pt serif, line
length 18–28 picas — the readability fundamentals).

The mistake is to vary the wrong thing at the wrong cadence. If the
masthead changes every issue, the newspaper has no identity. If the
lead-item treatment never changes, the newspaper has no rhythm.

### 1.3. Variation Serves Content, Not Itself

This is Brody's anti-style principle and it is operative. We do not
vary typography to demonstrate that we *can*. We vary because a
particular issue is feature-heavy, or brief-heavy, or thread-heavy,
or because a story warrants amplification, or because a page needs
breath. The system must be *responsive*: the layout reflects what
the curation produced this week. A week with one massive 4000-word
feature looks different from a week with twenty 200-word briefs,
because the typography responds to the content.

---

## 2. The Unit of Typographic Identity

Before listing dimensions, fix the units. Below, "edition" means
one published issue of one masthead. The architect should treat
these as nested levels:

| Level | What lives here | Cadence of variation |
|---|---|---|
| **Masthead** | The newspaper's name, identity, base type system, color/B&W posture | Settled per masthead. Changes with intention, not weekly. |
| **Edition** | One issue. Date, lead story, this week's section emphasis | Each issue selects from the masthead's allowed range. |
| **Section** | Features, In Brief, threads, etc. | Selected within the edition; consistent across an edition. |
| **Page** | One physical page. Lead item gets differential treatment. | Composed by row-packer; lead is identified per page. |
| **Article** | One item. Standard, feature, brief, thread, pull quote. | Treatment chosen by template + layout_hint + tier. |

The architect needs all five levels. Today the system has only one
level (article-template selection) and exposes nothing above it.

---

## 3. The Dimensions of Variation

Each dimension below has the same structure:
- **What it is.** The typographic property.
- **Range.** The spectrum, with concrete values.
- **What good looks like.** The principle.
- **What bad looks like.** The failure mode.
- **Cadence.** Edition / section / page / article.
- **Interactions.** Other dimensions this constrains or
  is constrained by.

### 3.1. Type Family System

**What it is.** The font families used and how they are paired.

**Range.** I propose three named systems for OffScroll. Each is a
*system*, not a font; each makes internally coherent decisions.

- **System A — "Serif-Forward Classical."**
  Body: Source Serif 4 Regular (10pt). Headlines: Source Serif 4
  Semibold/Bold. Metadata: Source Sans 3 Regular small caps. The
  sans is a quiet supporting voice. Reading register: serious, slow,
  long-form. Reference points: *NYRB*, *The Atlantic* print, *The
  Guardian Long Read*.

- **System B — "Sans-Headline Modern" (current default).**
  Body: Source Serif 4 Regular. Headlines: Source Sans 3 Bold.
  Metadata: Source Sans 3. The headline-body contrast is the
  primary expressive axis. Reading register: contemporary,
  newspaper-of-record. Reference points: *NYT*, *FT*, *Le Monde*.

- **System C — "Editorial Display."**
  Body: Source Serif 4 Regular. Headlines: a display serif
  (candidate: *EB Garamond Bold* or *Source Serif 4 Black*) with
  generous tracking variation. Metadata: small-caps Source Sans.
  Reading register: magazine-leaning, feature-heavy. Reference
  points: *Harper's*, *The New Yorker* feature openers.

(System D — "Condensed/Tabloid" — held in reserve for a future
masthead variant. Don't build it yet.)

**What good looks like.** Each system is internally consistent
across an entire edition. A reader picking up the issue
immediately recognizes which system they're holding because the
type voice is unified across mast, section heads, headlines,
body, metadata, and captions.

**What bad looks like.** Systems pollute each other. A System A
edition with a System B headline is a mistake. A "blend" is not a
fourth system; it is a broken one of the three.

**Cadence.** Per masthead, OR per edition only when the masthead
is conceived as a multi-system identity. For OffScroll's first
production cohort, treat type family system as **per masthead** —
selected once and applied consistently. Multi-system mastheads are
a Phase 2 question.

**Interactions.** Constrains nearly everything below. Scale
ratios, rule weights, drop cap style, kicker style, and pull-quote
treatment all defer to the chosen system.

### 3.2. Type Scale (Headline Hierarchy)

**What it is.** The set of point sizes used for body, byline,
caption, brief, standard headline, thread headline, feature
headline, masthead, section label, kicker.

**Range.** Three named scales. The body anchor is fixed
(9.5–10.5pt); ratios above are what vary.

- **Tight scale (compressed hierarchy).** Body 10pt, standard
  headline 13pt, thread 16pt, feature 22pt, masthead 36pt. Ratios
  approx 1.3 / 1.6 / 2.2 / 3.6. Register: dense, information-rich,
  many-stories-per-page. Reference: *FT*, *NYT* daily.

- **Standard scale (current default).** Body 10pt, standard 14pt,
  thread 18pt, feature 28pt, masthead 48pt. Ratios approx 1.4 /
  1.8 / 2.8 / 4.8.

- **Open scale (dramatic hierarchy).** Body 10pt, standard 16pt,
  thread 22pt, feature 36pt, masthead 60pt. Ratios approx 1.6 /
  2.2 / 3.6 / 6.0. Register: feature-heavy, fewer stories,
  magazine-leaning.

**What good looks like.** The scale is internally consistent —
ratios between adjacent levels are similar (a tight scale stays
tight; an open scale stays open). Hierarchy is *immediately*
parseable: feature > thread > standard > brief, never ambiguous.

**What bad looks like.** Headline of one size is barely larger
than a deck of another size. Two adjacent levels at ratios <1.2
read as the same level. Worst: a tight body+standard ratio
combined with a dramatic feature+thread ratio (the eye reads the
feature as another publication's intrusion).

**Cadence.** **Per edition.** A given issue picks one scale and
holds it. Variations within an edition come from *which* level
each item gets, not from inflating one level's size.

**Interactions.** With column count (open scales need fewer
columns, more white space). With weight (heavier weight = can
afford smaller size). With white-space rhythm (open scales need
generous spacing or they look top-heavy). With pull-quote sizing
(pull quote should sit in the gap between standard and thread,
proportional to the active scale).

### 3.3. Weight and Style Variation

**What it is.** Which weights (Light, Regular, Semibold, Bold,
Black) and styles (Roman, Italic, Small Caps) are used for which
elements.

**Range.** The current system uses two weights (Regular, Bold) and
two styles (Roman, Italic). This is too thin. Real editorial
typography uses 3–5 weights and exploits italic and small caps as
distinct registers, not just emphasis.

Recommended weight strategies:

- **Two-weight strategy (current).** Bold for headlines, Regular
  for body. Cleanest. Loses the deck-as-distinct-register move.
  Use only with tight or standard scales.

- **Three-weight strategy.** Light or Semibold introduced for
  decks/kickers/secondary headlines. Bold for primary headlines,
  Regular for body, Light or Semibold for the third register.
  Most expressive without becoming unstable.

- **Four-weight strategy.** Light + Regular + Semibold + Bold.
  Use when the masthead's personality demands typographic
  amplitude (System C, "Editorial Display"). Risky — easy to look
  like every weight is shouting.

Style usage:

- **Italic body insertions** — for editorial notes, captions,
  attribution. Currently used; treatment is fine.
- **Small caps for kickers and section labels.** Currently
  faked with `upper(...)`. This is wrong. Use real small caps
  (Source Sans 3 supports them via OpenType `smcp`). The
  difference is large in print. Configure this even if no other
  variation lands.
- **Italic decks.** A deck in Source Serif 4 Italic at 10–11pt
  reads as editorial voice, distinct from headline (information)
  and body (the article). Currently used on features only.
  Extend to standards in the open scale.
- **Light + tracking for kickers.** "COVER STORY" set in 9pt
  light sans with +0.08em tracking is a different register from
  bold-uppercase. Both have valid places.

**What good looks like.** Each register (headline / deck / byline
/ kicker / body / caption / pull quote / metadata) has a single,
recognizable typographic identity that does not collide with any
other register. A reader can identify which register they're
reading without consciously parsing it.

**What bad looks like.** Bold used for everything that wants
emphasis (headline, kicker, byline, pull quote, caption). The
result is a page that looks "shouty" with no actual hierarchy —
bold loses its meaning when overused. Equally bad: italic used as
a decorative finish on captions *and* decks *and* pull quotes
*and* attribution. Italic also loses meaning when overused.

**Cadence.** **Per edition** for the weight strategy (which set of
weights is in play). **Per article** for which weight a specific
element gets — but constrained to the strategy chosen at the
edition level.

**Interactions.** With type family system. With scale (heavier
weights work at smaller sizes; lighter weights need more size).
With column width (light weights at narrow column widths get
fragile).

### 3.4. Lead-Item Differentiation

**What it is.** How the top story on a given page distinguishes
itself from the rest of that page. This is the single most
important dimension for breaking the current monotony, because
right now *only the front-page feature* gets differential
treatment. Pages 2 through N are visually flat — every standard
article looks the same.

**Range.** The lead on each interior page should employ one of
the following amplifications, chosen by the architect from a
small set:

- **Scale amplification.** Lead headline one step up from the
  standard size on that page (e.g., 18pt where the others are
  14pt).
- **Deck added.** Lead gets a 1–2 line italic deck; others do
  not. Decks are currently feature-only — extending them to leads
  is a powerful and cheap differentiator.
- **Drop cap.** Lead gets a drop cap on the first paragraph;
  others do not. Currently feature-only.
- **Hairline anchor rule.** A 1pt rule above the lead headline,
  spanning its column width, anchors the lead visually to the
  page top.
- **Wider column allocation.** Lead occupies 2 of 3 columns
  while siblings occupy 1; the lead's body wraps in 2 columns
  while others are single-column.
- **Image priority.** If images are present, the lead claims
  the largest one. Others get smaller or none.

A page should use **one or two** lead-amplification moves, not
all six. The goal is unmistakable hierarchy with minimum
typographic noise.

**What good looks like.** Within two seconds of seeing a page,
the reader knows which story is the lead. The amplification feels
proportional to the story's importance, not gratuitous.

**What bad looks like.** Every page mimics the front feature
(everything has a drop cap and a hero image). Or the inverse:
nothing is amplified, and the page reads as a list of
indistinguishable items. The current system errs hard toward the
latter.

**Cadence.** **Per page.** A page has one lead. The
amplification choice can be per-edition (every page uses the
same amplification recipe in this issue) or per-page (different
pages amplify differently). I recommend **per-edition** for
launch — a consistent rhythm across pages — and per-page as a
later refinement.

**Interactions.** With section structure (a page that opens a
new section may use the section label as part of the lead
amplification). With image availability (image priority only
works when images exist). With scale (the size step must be at
least 1.3× to register as differentiation, not just slop).

### 3.5. Pull Quote Treatment

**What it is.** How extracted quotations are visually rendered.

**Range.** Five treatments, each with editorial connotations:

- **Hairline-bordered (current).** Top and bottom 1pt rules, 14pt
  italic, 85% width, centered. Quiet, classical, magazine-modest.

- **Bold-displayed.** No rules; very large (24–32pt) sans-bold
  quote, hung at the column edge, no quotation marks. Loud,
  contemporary. Reference: *Wired* feature openers.

- **Slab-quoted.** A large opening quotation mark in display size
  (40–60pt) followed by italic body-size quote. Old-school,
  literary. Reference: *NYRB*.

- **Marginalia.** Pull quote set in a margin column, 9pt italic
  with a hairline rule. Decorative but quiet. Requires a layout
  with visible margin space (asymmetric grid).

- **Indented italic, no rule.** Just an indent + italic + slightly
  reduced size. Most restrained. Useful when the article is
  already typographically loud and another rule would overwhelm.

**What good looks like.** The treatment fits the article's
register. Quote pulled from a serious investigative piece gets a
hairline-bordered or slab-quoted treatment. Quote pulled from a
tech feature gets bold-displayed. Treatments are not mixed within
an edition.

**What bad looks like.** Same treatment for every pull quote
across editions of every register — which is what we have now.
Or: multiple treatments mixed within a single edition (looks
broken).

**Cadence.** **Per edition** for which treatment is active.
Optional: **per article** for the position (inline vs.
between-article) — this is already handled in the row composer.

**Interactions.** With type family system (System A favors
slab-quoted or hairline; System C favors bold-displayed or
marginalia). With pull-quote frequency (loud treatments work only
when quotes are scarce; if every long article has a pull quote,
loud treatment becomes oppressive).

### 3.6. White-Space Rhythm

**What it is.** The breathing of the page. Inter-paragraph
spacing, space above/below headlines, around section breaks,
between articles, page margins.

**Range.** I propose three named rhythms, all derived from a
single base unit per edition (the "metric step").

- **Tight rhythm.** Metric step = 0.04in. Inter-paragraph 0.04in,
  above-headline 0.10in, section break 0.18in, page margin
  0.4in. Register: dense, information-heavy, brief-rich.

- **Standard rhythm (current default).** Metric step = 0.05in.
  Inter-paragraph 0.05in, above-headline 0.15in, section break
  0.25in, page margin 0.5in.

- **Open rhythm.** Metric step = 0.07in. Inter-paragraph 0.07in,
  above-headline 0.20in, section break 0.35in, page margin
  0.6in. Register: feature-heavy, magazine-leaning.

**What good looks like.** The rhythm is consistent across the
edition — a reader who enters a section feels the same breathing
they felt on the previous page. Section breaks feel more
substantial than article breaks; article breaks feel more
substantial than paragraph breaks. The hierarchy of spaces
matches the hierarchy of content.

**What bad looks like.** Article breaks the same size as
paragraph breaks (no perceived structure). Section breaks the
same size as article breaks (no perceived sections). Or the
inverse — section breaks so large the page reads as sparse.

**Cadence.** **Per edition.** All three rhythms must scale
together — you can't open up the section breaks while keeping
inter-paragraph tight; the proportions break.

**Interactions.** With type scale (open scales need open
rhythms; tight scales tolerate tight rhythms). With column count
(narrow columns need slightly more inter-paragraph space; wide
columns can tolerate tighter). With page margin (an open rhythm
demands generous page margins or the page looks unbalanced).

### 3.7. Grid Variation

**What it is.** Number of columns, gutter width, and column-width
ratios across the page.

**Range.** Brody's principle stands: simple grids, complex
execution. I propose three grid systems, each used at the page
or row level.

- **Three-column equal (current default).** Three equal columns,
  0.25in gutter. The workhorse. Use for 70% of pages.

- **Two-column equal.** Two equal columns, 0.30in gutter. Use
  for feature-heavy pages where article density is low and each
  story needs more breathing room. Wider columns mean longer
  measure (~32 picas) which suits long-form serif body.

- **Asymmetric 2:1 (Swiss-derived).** A wide column (~2x) plus a
  narrow column (~1x), 0.25in gutter. The wide column hosts the
  primary article body. The narrow column hosts marginalia,
  small images, briefs, or pull quotes. Use sparingly — one or
  two pages per edition — as a deliberate compositional event.

A four- or five- or six-column grid is **not** recommended.
Brody is right: those are engineering solutions that produce
uniform results. The three options above produce more variety
than is currently being achieved with one option.

**What good looks like.** Within an edition, 80–100% of pages
use one grid (the edition's "home" grid). One or two pages may
deploy a different grid as a compositional event. The
asymmetric grid is used when the content demands it (a long
feature with marginalia, a brief-heavy page with image
sidebars), not as decoration.

**What bad looks like.** Grid changes per article on the same
page. Six-column grid used to "be flexible" — produces uniform
results because the columns are too narrow to host meaningful
variation. Asymmetric grid used for body content with no
material for the narrow column (the narrow column ends up empty
or padded with throwaway items).

**Cadence.** **Per edition** for the home grid. **Per page**
for grid deviations. Within a single page, one grid only.

**Interactions.** With type scale (open scale + 3-column =
crowded; open scale + 2-column = correct). With column gap
(wider columns can tolerate slightly tighter gaps; narrow
columns need more gap to avoid eye-bridging). With pull-quote
treatment (marginalia treatment requires asymmetric grid).

### 3.8. Image and Figure Treatment

**What it is.** How images sit within the grid.

**Range.** Five treatments:

- **Column-width image.** Image fits one column, breaks for
  caption. The default. Quiet, integrated.
- **Multi-column image.** Image spans 2 of 3 columns, body text
  wraps below. For lead images and feature openers.
- **Full-page-width image.** Image spans all columns; body
  flows below or above. For dramatic feature openers.
- **Bleed image (future).** Image bleeds off page edge. Brody
  principle: "bleeding beyond the frame suggests there's more
  to see outside." Requires print-bleed margins (currently 0in
  bleed; would require pipeline work). Worth aspiring to.
- **Square-cropped image.** Native aspect overridden to 1:1.
  Imposes typographic unity but distorts photography. Use only
  for headshots or graphic elements, never for editorial photos.

**What good looks like.** Image treatment matches the article's
weight. Lead/feature gets multi-column or full-width. Standard
gets column-width. Brief gets none or a small column-width
crop. Captions are 8pt sans, italic optional, attributed.

**What bad looks like.** Every image full-width (page reads as
photo essay; text loses primacy). Every image column-width
(every page reads the same; lead has no image priority). Caption
typography mixed (some sans, some serif, some italic) within
the same edition.

**Cadence.** **Per article** for treatment choice, constrained
by the article's tier (feature/standard/brief). **Per edition**
for caption typography.

**Interactions.** With lead-item differentiation (image priority
is one of the lead amplifications). With grid (full-width images
require all-columns spanning capability). With white-space
rhythm (large images need open rhythm or the page suffocates).

### 3.9. Color and Rule Usage

**What it is.** The palette (or absence of palette) and the
treatment of horizontal/vertical rules.

**Range.** OffScroll is fundamentally a B&W product (ink
economy, home-printer compatibility). But "B&W" has range:

- **Pure B&W (current).** Two values: text (luma 26) and rule
  (luma 153–238, used for hairline structure). Most ink-economical.

- **B&W + one accent.** A single accent color (a deep red, a
  navy, or a forest green) used sparingly: section-label color,
  a hairline anchor rule above the lead, the masthead bar.
  Ink cost is small if used judiciously. Adds a publication
  identity without overwhelming.

- **Single-tone gray scale.** A 10–20% gray tint block behind a
  pull quote or section header, otherwise pure B&W. No accent
  color. Ink cost is moderate (tints consume ink). Works on
  laser; problematic on inkjet.

Rule treatments:

- **Hairline rules** (0.5pt, luma 200–238) — separators between
  briefs, beneath bylines, between row columns.
- **Standard rules** (0.5pt, luma 26) — beneath section labels,
  framing pull quotes.
- **Heavy rules** (2–3pt, luma 26) — masthead underline, major
  section openers.
- **Double rules** (a 1pt + 0.25pt + 1pt sandwich) — a classical
  newspaper move, used at most once per edition for the masthead
  or front-feature anchor. Distinctive without being loud.

**What good looks like.** Rule weights map to hierarchy
(hairline = fine separator, heavy = section opener). Color use is
restrained and meaningful. The B&W system is the default; accent
color is earned, not sprayed.

**What bad looks like.** Tint blocks behind every section header
(ink hog, looks 1990s). Multiple accent colors (loses identity).
Rule weights chosen inconsistently (some sections use hairlines,
others use 1pt rules, with no logic).

**Cadence.** **Per masthead** for color posture (B&W only,
B&W+accent, or grayscale). **Per edition** for accent color use
intensity (which elements wear it this issue).

**Interactions.** With ink economy (a hard constraint —
masthead must be printable on a $30 inkjet). With type family
system (System C tolerates more decorative rules; System A
prefers restraint).

### 3.10. Section Header Treatment

**What it is.** How "Features," "In Brief," and other section
labels are rendered.

**Range.** Currently: 14pt bold sans uppercase, 2pt rule above,
sticky. This is fine but uniform. Three treatments to choose
from:

- **Heavy-rule label (current).** 14pt bold sans uppercase,
  2pt rule above. Strong sectioning, news-paper register.
- **Hairline label.** 11pt small-caps, 0.5pt rule above and
  below (sandwich). Quieter, magazine register.
- **Display label.** 22pt italic serif (no rule, large white
  space above). Loud, editorial-feature register. For mastheads
  using System C.

**What good looks like.** Sectioning is unambiguous —
the eye knows it has crossed into new territory. The treatment
matches the type family system.

**What bad looks like.** Section header indistinguishable from
a feature headline (hierarchy collapses). Section header so loud
it dominates the section's actual content.

**Cadence.** **Per edition** within the masthead's allowed
range.

**Interactions.** With type family system (must match).
With white-space rhythm (display label requires open rhythm).

### 3.11. Drop Cap and Lead-Paragraph Treatment

**What it is.** How the opening of a long article is signaled.

**Range.**

- **Drop cap, sans bold (current).** 36pt sans bold initial,
  3-line drop, 0.06in right inset. Modern, contemporary.
- **Drop cap, serif italic.** 42pt serif italic initial,
  3-line drop. Traditional, literary. Pairs with System A.
- **Raised cap.** Initial sits *above* the baseline of line one,
  in display size (24pt sans or serif). Less formal than a drop
  cap, used for shorter articles.
- **No cap, lead in small caps.** The first 3–6 words set in
  small caps. Quiet alternative to a cap. Pairs well with System
  A and tight scales.
- **No special treatment.** Plain first paragraph. Default for
  briefs and short standards.

**What good looks like.** The treatment marks the article as
"long enough to reward extended reading." Drop caps appear
*only* on items that earn them — features and the lead standard
on each page, not every article.

**What bad looks like.** Drop caps everywhere (loses meaning).
Drop cap font misaligned with system (a sans drop cap in a
serif-forward edition looks like a graft). Drop cap height
larger than the deck or kicker above it (hierarchy inverts).

**Cadence.** **Per edition** for the cap style. **Per article**
for whether a cap is used.

**Interactions.** With type family system. With lead-item
differentiation (drop cap may be the chosen amplification).
With word count (drop caps need a body of at least ~500 words
to feel earned).

### 3.12. Byline and Kicker Treatment

**What it is.** How metadata above and below headlines is set.

**Range.**

- **Italic gray byline (current).** 9pt sans italic, luma 102.
  Quiet, contemporary.
- **Small-caps byline.** 8pt sans small caps, regular weight,
  luma 102. Quieter still, more classical.
- **Slug-byline.** Author name in upper-case + tracking, then a
  thin rule, then source. More structured.
- **Bold-author byline.** Author bold sans, source italic
  serif. More emphasis on the writer; suits opinion pieces.

Kicker (the "COVER STORY" / section label above the headline):

- **Bold uppercase (current).** 9pt bold sans, uppercase.
  Strong section identity.
- **Light tracking.** 9pt light sans, uppercase, +0.08em tracking.
  Quieter, more classical.
- **Small caps with rule.** 9pt sans small caps with a 0.5pt rule
  above. Magazine register.

**What good looks like.** Byline and kicker form a coherent
metadata register. They share family, weight family (both light
or both bold), and color (luma 102–153). They do not compete with
the headline.

**What bad looks like.** Kicker bolder than headline (hierarchy
inverts). Byline so quiet it disappears (attribution is
editorial respect; making it invisible is a choice with
consequences). Mixed treatments within an edition.

**Cadence.** **Per edition** within the system's allowed range.

**Interactions.** With type family system. With weight strategy.

### 3.13. Masthead Treatment

**What it is.** The newspaper's nameplate at the top of page 1.

**Range.** I am cautious here. The masthead is the single most
identity-defining element. It should *not* vary edition to
edition. It is the constant that signals "this is your
newspaper."

But the masthead's treatment is still a design decision per
masthead, and OffScroll generates many mastheads, so it must be
parameterizable.

- **Centered classical.** Large bold sans, centered, hairline
  rules above and below. Current default. Identity: dignified,
  newspaper-of-record.
- **Display serif, centered.** 56pt serif (Source Serif 4
  Black), centered. Identity: literary, magazine-leaning. Pairs
  with System A or C.
- **Asymmetric flag.** Masthead set flush-left, with date and
  volume flush-right on the same line. Identity: contemporary,
  alternative-press.
- **Wordmark + tagline.** Masthead 36pt with a tagline below in
  small caps. Identity: branded, almost magazine.

**What good looks like.** The masthead is *the* moment of
typographic identity. It should be unambiguous, distinctive, and
held constant across an edition cohort. Variation between
mastheads is desired; variation within a masthead is not.

**What bad looks like.** Masthead treatment that looks like
every other newspaper (default centered bold sans, no
distinctiveness). Or masthead that changes per issue (newspaper
loses identity).

**Cadence.** **Per masthead.** Locked at masthead creation.

**Interactions.** With type family system (must match).
With color posture (mastheads using accent color claim it as
identity).

### 3.14. Footer and Running Elements

**What it is.** Page footer, page number, running head if any.

**Range.** Current implementation is a single line with title
and date, centered. Acceptable but could vary:

- **Centered single-line (current).** Title em-dash date, 7pt
  sans light gray.
- **Asymmetric.** Page number flush-left, title flush-right,
  hairline rule above.
- **None.** No footer. Page number only, in the outer corner.
  Quiet, classical.

**Cadence.** **Per masthead.** Settled with the identity.

**What good looks like.** Footer is a quiet structural marker.
It should never compete for attention. Page numbers always exist
(except optionally on page 1).

**What bad looks like.** Footer same point size and color as a
caption (visual confusion). Footer absent (reader navigation
suffers in long editions).

---

## 4. Interaction Constraints (the matrix)

Variation must be coherent. The following pairs are
**incompatible**:

- **Open scale + 3-column equal grid + tight rhythm.** The
  feature headline is 36pt; the column width is too narrow to
  host it cleanly. Pick two: open scale + 2-column grid + open
  rhythm.
- **Tight scale + 2-column equal grid + open rhythm.** Sparse
  page; too much white, headlines too small to anchor it.
- **System A (Serif-Forward Classical) + bold-displayed pull
  quote.** Voice mismatch. System A favors hairline-bordered
  or slab-quoted.
- **System C (Editorial Display) + tight rhythm.** Display
  scale demands breath.
- **Display label section header + tight rhythm.** Same reason.
- **Two-weight strategy + small caps for kickers.** Small caps
  are a third register; you've effectively gone to three.
- **Pure B&W + grayscale tint blocks.** Pick one color posture.

The following pairs are **complementary**:

- Tight scale + 3-column + tight rhythm + System B.
- Standard scale + 3-column + standard rhythm + System B.
  (Current defaults — coherent.)
- Open scale + 2-column + open rhythm + System A or C.
- Asymmetric grid + marginalia pull quote.
- Display label section header + System C + open rhythm.

The architect should not expose 12 independent dials. The
architect should expose a small number of *presets* (coherent
bundles) plus a few orthogonal axes (e.g., section color accent
intensity, lead amplification choice) that can vary on top.

---

## 5. Implementation Hierarchy: What to Build First

Not all dimensions are equally valuable. Here is my prioritized
order. Build top to bottom.

**Tier 1 — must-have. Without these, the corpus stays
homogeneous.**

1. **Type scale (3.2).** Three named scales, edition-level
   selection. This is the single highest-leverage variation.
2. **Lead-item differentiation (3.4).** Per-page lead
   amplification using deck or scale step. Currently the page
   structure is flat after the front feature.
3. **Weight strategy (3.3).** Move from two weights to three.
   Real small caps for kickers. Italic decks on standards (not
   just features).

**Tier 2 — high value. Adds editorial range.**

4. **White-space rhythm (3.6).** Three named rhythms, locked to
   scale. Without rhythm matching scale, scale variation looks
   broken.
5. **Drop cap and lead-paragraph treatment (3.11).** Three cap
   styles + small-caps lead-in. Per edition.
6. **Pull quote treatment (3.5).** Five treatments. Per edition.

**Tier 3 — adds identity. Per masthead.**

7. **Type family system (3.1).** A, B, C. Three systems.
8. **Section header treatment (3.10).**
9. **Masthead treatment (3.13).**
10. **Color and rule usage (3.9).** Including the optional
    accent color.

**Tier 4 — refinements. Build only after Tiers 1–3 ship.**

11. **Grid variation (3.7).** Two- and three-column home grids;
    asymmetric grid for special pages.
12. **Image and figure treatment (3.8).** Multi-column and
    full-width images for leads.
13. **Byline and kicker treatment (3.12).**
14. **Footer and running elements (3.14).**

If the architect can build only Tier 1 in the first sprint, that
is sufficient to break the homogeneity of the corpus. The
remaining tiers are value-add, not table stakes.

---

## 6. The Variation Sampling Problem (a note for Ada)

Once these dimensions exist, the system needs to sample over
them when generating the training corpus. Two principles for
that sampling:

1. **Sample bundles, not parameters.** The training corpus
   should contain editions that are coherent within themselves
   and varied between themselves. Don't sample each parameter
   independently — that produces broken pages. Sample a
   *preset bundle* (e.g., "open + 2-col + open rhythm + System
   A + slab pull quote + serif drop cap") and apply consistently
   across an edition.

2. **Stratify across the design space.** If there are, say, 6
   coherent presets and 200 training editions, ensure roughly
   30 editions per preset. Don't let the sampler concentrate on
   one preset because of conditional probability accidents.

The dimensions described above naturally yield a tractable
preset count. Three scales × three rhythms × two grids × three
systems = 54 raw combinations, but the interaction constraints
(§4) cut this to ~12 coherent presets. Plus per-page lead
amplification (~5 variants) gives ~60 distinct typographic
states a page can occupy. That is the right order of magnitude
for a model trained on 200–500 examples to learn from.

---

## 7. Anti-Patterns: What This Document Is Not Asking For

I want to be specific about what I am *not* recommending,
because the natural over-reaction to "increase typographic
diversity" is to produce chaos.

- **Not** randomized parameters per article. That produces
  broken pages.
- **Not** more font families. Three Source families is
  sufficient. The expressive range comes from how they're
  deployed, not from importing more fonts.
- **Not** more grid columns. Three columns is the maximum.
  Brody is right: complexity should come from composition, not
  from grid columns.
- **Not** decorative ornaments (dingbats, fleurons, decorative
  rules with curlicues). These are not editorial; they are
  costume.
- **Not** color photographs at full bleed. The product is B&W
  print on home printers. Treat color as a Phase 2 question.
- **Not** novelty fonts for headlines. Source Sans/Serif at
  varied weights and sizes is sufficient for editorial range.
  A "playful" headline font is not editorial design; it is
  branding theater.

The variation I am asking for is *editorial range within a
coherent system*. Not visual chaos; not decorative
ornamentation; not gimmicks. Real editorial typography varies
because real content varies, and the variation is governed by
principles a designer can defend.

---

## 8. Closing Position

The current system is monotonal. The proposed system is
tonal — multiple voices, each internally consistent, each
deployed when the content asks for it. This is what makes a
newspaper worth holding in print. It is also what gives the ML
grader something real to learn.

The architect should treat this document as the design
specification for the next version of the template system. Where
specific values are needed (point sizes, gutters, weights), I
have provided them. Where judgment calls remain, I am available
for review of the implementation against intent.

The proof of this work will be in print. Print three editions,
each from a different preset bundle, on a B&W laser printer, and
hold them next to each other on a desk. If they feel like three
different newspapers — each coherent, each appropriate — the
system works. If they feel like the same newspaper with knobs
twiddled, the system has failed and we iterate.

— Neville

COMPLETED
