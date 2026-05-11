// OffScroll Typst Templates
// Article-type rendering functions for newspaper layout.
// These are imported by newspaper.typ and called by the Python
// markup builder via data substitution.
//
// Typographic diversity (Neville §3.2, §3.3, §3.4 — Tier 1):
// The template system is parameterised by an *edition config* carried
// in Typst state. A scale preset selects the headline hierarchy for the
// whole edition; a weight preset selects the register strategy; lead
// amplifications determine how the first item on each section gets
// differentiated. The Python builder emits #set-edition-config(...) at
// the top of the generated file. Defaults reproduce the prior layout
// (standard scale) but always activate small-caps kickers, which the
// design spec asks for unconditionally.

// --- Static design tokens (font families, colours, gutters) ---
#let body-font = "Source Serif 4"
#let sans-font = "Source Sans 3"
#let mono-font = "Source Code Pro"
#let column-gap = 0.25in
#let rule-stroke = 0.5pt + luma(221)
#let dark-rule = 2pt + luma(26)
#let light-rule = 0.5pt + luma(204)
#let text-color = luma(26)
#let meta-color = luma(102)
#let light-meta = luma(153)
#let metadata-size = 9pt
#let caption-size = 8pt
#let brief-size = 9pt

// --- Scale presets (Neville §3.2) ---
// Body is fixed at 10pt across all scales — the readability anchor.
// Sizes above scale with the chosen edition personality.
#let scale-presets = (
  tight: (
    body: 10pt,
    standard: 13pt,
    thread: 16pt,
    feature: 22pt,
    masthead: 36pt,
    deck: 10pt,
    pull-quote: 12pt,
    section: 11pt,
    drop-cap: 30pt,
    drop-cap-baseline: 19pt,
    lead-bump: 16pt,   // standard headline when amplified as lead (=thread)
  ),
  standard: (
    body: 10pt,
    standard: 14pt,
    thread: 18pt,
    feature: 28pt,
    masthead: 48pt,
    deck: 10pt,
    pull-quote: 14pt,
    section: 14pt,
    drop-cap: 36pt,
    drop-cap-baseline: 22pt,
    lead-bump: 18pt,
  ),
  open: (
    body: 10pt,
    standard: 16pt,
    thread: 22pt,
    feature: 36pt,
    masthead: 60pt,
    deck: 11pt,
    pull-quote: 16pt,
    section: 16pt,
    drop-cap: 44pt,
    drop-cap-baseline: 26pt,
    lead-bump: 22pt,
  ),
)

// --- Weight presets (Neville §3.3) ---
// "two" preserves the prior two-register treatment (bold + regular,
// uppercase kickers). "three" introduces a third register via Source
// Sans 3 Light + real small caps for kickers, plus italic decks broadly
// available to standard articles, not just features.
#let weight-presets = (
  two: (
    headline-weight: "bold",
    kicker-weight: "bold",
    kicker-style: "upper",
    kicker-tracking: 0em,
    deck-style: "italic",
    deck-applies-to-standards: false,
  ),
  three: (
    headline-weight: "bold",
    kicker-weight: "light",
    kicker-style: "smallcaps",
    kicker-tracking: 0.05em,
    deck-style: "italic",
    deck-applies-to-standards: true,
  ),
)

// --- Edition config state ---
// Carried through the document; written by the generated .typ file at
// the top and read by every template via #context.
#let edition-config-default = (
  scale: "standard",
  weights: "three",
  lead-amplifications: ("deck", "scale-bump"),
)
#let edition-config = state("offscroll-edition-config", edition-config-default)

#let set-edition-config(
  scale: "standard",
  weights: "three",
  lead-amplifications: ("deck", "scale-bump"),
) = {
  edition-config.update((
    scale: scale,
    weights: weights,
    lead-amplifications: lead-amplifications,
  ))
}

// --- Accessors (must be evaluated inside #context) ---
#let current-sizes() = {
  let cfg = edition-config.get()
  let key = if cfg.scale in scale-presets { cfg.scale } else { "standard" }
  scale-presets.at(key)
}
#let current-weights() = {
  let cfg = edition-config.get()
  let key = if cfg.weights in weight-presets { cfg.weights } else { "three" }
  weight-presets.at(key)
}
#let lead-amps() = edition-config.get().lead-amplifications

// --- Kicker renderer (style-aware) ---
// Renders the kicker text in the active edition's kicker style.
// Called from inside #context blocks so weights/sizes are current.
#let kicker-block(kicker-text, sizes, weights, color: meta-color) = {
  let style = weights.kicker-style
  let weight = weights.kicker-weight
  let tracking = weights.kicker-tracking
  set text(metadata-size, font: sans-font, weight: weight,
           tracking: tracking, fill: color)
  if style == "smallcaps" {
    smallcaps(kicker-text)
  } else {
    upper(kicker-text)
  }
}

// --- Deck renderer ---
// Italic serif deck. Used by features unconditionally and by standards
// when the active edition's weight strategy enables decks for standards
// AND the article is marked as the page lead.
#let deck-block(deck-text, sizes, color: luma(68)) = {
  set text(sizes.deck, font: body-font, style: "italic", fill: color)
  set par(leading: 0.52em)
  deck-text
}

// --- Pull Quote ---
#let pull-quote(text-content, attribution) = context {
  let sizes = current-sizes()
  block(breakable: false, above: 0.08in, below: 0.08in,
    align(center,
      block(width: 85%, stroke: (top: 1pt + text-color, bottom: 1pt + text-color),
        inset: (y: 0.08in))[
        #set text(sizes.pull-quote, font: body-font, style: "italic")
        #set par(leading: 0.48em)
        #text-content
        #v(0.04in)
        #set text(metadata-size, font: sans-font, style: "normal", fill: meta-color)
        #attribution
      ]
    )
  )
}

// --- Image Block ---
#let image-block(img-path, caption-text: none) = {
  block(breakable: false, above: 0.05in, below: 0.08in, clip: true)[
    #if img-path != none and img-path != "" {
      block(clip: true, height: auto)[
        #image(img-path, width: 100%, fit: "contain")
      ]
    }
    #if caption-text != none and caption-text != "" {
      v(0.03in)
      text(caption-size, font: sans-font, fill: meta-color)[#caption-text]
    }
  ]
}

// --- Drop Cap (scale-aware) ---
#let drop-cap(letter) = context {
  let sizes = current-sizes()
  box(
    baseline: sizes.drop-cap-baseline,
    inset: (right: 0.06in, top: 0.02in),
    text(sizes.drop-cap, weight: "bold", font: sans-font, fill: text-color)[#letter]
  )
}

// --- Feature Article ---
#let feature-article(
  title: none,
  kicker: "Cover Story",
  author: "",
  source-name: none,
  hero-image: none,
  hero-caption: none,
  deck: none,
  lead-pre: none,
  lead-cap: none,
  lead-rest: none,
  body-paragraphs: (),
  inline-pq: none,
  inline-pq-idx: -1,
  edited-for-length: false,
) = context {
  let sizes = current-sizes()
  let weights = current-weights()

  block(breakable: true, below: 0.2in, stroke: (bottom: 0.5pt + luma(204)),
    inset: (bottom: 0.15in))[
    // Hero image
    #if hero-image != none and hero-image != "" {
      block(clip: true)[
        #image(hero-image, width: 100%, fit: "contain")
      ]
      if hero-caption != none and hero-caption != "" {
        text(7pt, font: sans-font, fill: meta-color)[#hero-caption]
      }
      v(0.1in)
    }

    // Kicker (real small caps in three-weight strategy)
    #kicker-block(kicker, sizes, weights)
    #v(0.03in)

    // Title
    #if title != none {
      block(sticky: true)[
        #text(sizes.feature, weight: weights.headline-weight, font: sans-font)[
          #set par(leading: 0.4em)
          #title
        ]
      ]
      v(0.05in)
    }

    // Deck
    #if deck != none and deck != "" {
      deck-block(deck, sizes)
      v(0.08in)
    }

    // Byline
    #{
      set text(metadata-size, font: sans-font, style: "italic", fill: meta-color)
      author
      if source-name != none and source-name != "" and source-name != author {
        [ · #source-name]
      }
    }
    #v(0.06in)

    // Lead paragraph with drop cap
    #{
      set text(10.5pt)
      set par(leading: 0.5em, justify: true)
      if lead-cap != none {
        [#lead-pre#drop-cap(lead-cap)#lead-rest]
      }
    }
    #v(0.08in)

    // Body (2-column)
    #if body-paragraphs.len() > 0 {
      columns(2, gutter: column-gap)[
        #set par(justify: true)
        #set text(sizes.body, hyphenate: true)
        #for (idx, para) in body-paragraphs.enumerate() {
          [#para]
          v(0.05in)
          if idx == inline-pq-idx and inline-pq != none {
            inline-pq
          }
        }
      ]
    }

    // Edited for length
    #if edited-for-length {
      align(right, text(7pt, style: "italic", fill: light-meta)[(Edited for length)])
    }
  ]
}

// --- Standard Article ---
// is-lead: when true, the page lead receives configured amplifications
// (deck and/or scale-bump). Cf. Neville §3.4.
// deck: optional 1-sentence summary, only rendered when is-lead AND
// "deck" is in the edition's lead-amplifications AND the weight strategy
// permits decks on standards.
#let standard-article(
  title: none,
  author: "",
  source-name: none,
  images: (),
  paragraphs: (),
  insert-map: (:),
  inline-pq: none,
  inline-pq-idx: -1,
  word-count: 0,
  edited-for-length: false,
  editorial-note: none,
  debug-mode: false,
  is-lead: false,
  deck: none,
) = context {
  let sizes = current-sizes()
  let weights = current-weights()
  let amps = lead-amps()

  // Headline size: amplified for the page lead when "scale-bump" is active.
  let title-size = if is-lead and "scale-bump" in amps {
    sizes.lead-bump
  } else {
    sizes.standard
  }

  // Decide whether to render a deck.
  let show-deck = (
    is-lead
    and "deck" in amps
    and weights.deck-applies-to-standards
    and deck != none
    and deck != ""
  )

  block(breakable: word-count > 200, below: 0.15in)[
    // Headline
    #if title != none {
      block(sticky: true)[
        #text(title-size, weight: weights.headline-weight, font: sans-font)[
          #set par(leading: 0.4em)
          #title
        ]
      ]
      v(0.03in)
    }

    // Lead deck (italic, scale-aware)
    #if show-deck {
      deck-block(deck, sizes)
      v(0.05in)
    }

    // Byline
    #{
      set text(metadata-size, font: sans-font, style: "italic", fill: meta-color)
      author
      if source-name != none and source-name != "" and source-name != author {
        [ · #source-name]
      }
    }
    #v(0.05in)

    // First image
    #if images.len() > 0 {
      let img = images.at(0)
      image-block(img.at("path", default: ""), caption-text: img.at("caption", default: none))
    }

    // Body
    #block(breakable: word-count > 200)[
      #{
        let use-multicol = word-count > 200
        let body-content = {
          set par(justify: true)
          set text(sizes.body, hyphenate: true)
          let extra-images = if images.len() > 1 { images.slice(1) } else { () }
          for (idx, para) in paragraphs.enumerate() {
            [#para]
            v(0.05in)
            // Interleaved images
            let idx1 = idx + 1  // 1-based index matching Jinja loop.index
            if str(idx1) in insert-map {
              let img-idx = insert-map.at(str(idx1))
              if img-idx < extra-images.len() {
                let img = extra-images.at(img-idx)
                image-block(img.at("path", default: ""), caption-text: img.at("caption", default: none))
              }
            }
            // Inline pull quote
            if idx == inline-pq-idx and inline-pq != none {
              inline-pq
            }
          }
        }
        if use-multicol {
          columns(2, gutter: column-gap, body-content)
        } else {
          body-content
        }
      }
    ]

    // Edited for length
    #if edited-for-length {
      align(right, text(7pt, style: "italic", fill: light-meta)[(Edited for length)])
    }

    // Editorial note
    #if debug-mode and editorial-note != none {
      text(caption-size, style: "italic", fill: luma(85))[#editorial-note]
    }
  ]
}

// --- Thread ---
#let thread-article(
  headline: "",
  author: "",
  source-name: none,
  editorial-note: none,
  posts: (),
) = context {
  let sizes = current-sizes()
  let weights = current-weights()

  block(breakable: false, below: 0.15in)[
    // Headline
    #text(sizes.thread, weight: weights.headline-weight, font: sans-font)[
      #set par(leading: 0.4em)
      #headline
    ]
    #v(0.03in)

    // Byline
    #{
      set text(metadata-size, font: sans-font, style: "italic", fill: meta-color)
      author
      if source-name != none and source-name != "" {
        [ · #source-name]
      }
    }

    // Deck (editorial note for threads)
    #if editorial-note != none and editorial-note != "" {
      v(0.03in)
      block(stroke: (bottom: light-rule), inset: (bottom: 0.05in))[
        #text(metadata-size, font: body-font, style: "italic", fill: luma(68))[
          #set par(leading: 0.52em)
          #editorial-note
        ]
      ]
    }

    // Thread posts with left border
    #v(0.05in)
    #let total = posts.len()
    #block(stroke: (left: 2pt + luma(153)), inset: (left: 0.12in))[
      #for (idx, post) in posts.enumerate() {
        block(below: 0.08in)[
          #text(7pt, weight: "bold", font: sans-font, fill: luma(153))[
            #(idx + 1)/#total
          ]
          #v(0.02in)
          #set par(justify: true)
          #set text(sizes.body, hyphenate: true)
          #post
        ]
      }
    ]
  ]
}

// --- Brief Item ---
#let brief-item(author, source-name: none, text-content) = {
  block(below: 0.08in, stroke: (bottom: 0.5pt + luma(238)),
    inset: (bottom: 0.05in))[
    #set text(brief-size)
    #set par(leading: 0.52em)
    #text(weight: "bold")[#author#if source-name != none [, #source-name]:] #text-content
  ]
}

// --- Brief Group ---
// The "In Brief" label uses the active kicker treatment so the typographic
// register is consistent with article kickers in the same edition.
#let brief-group(briefs) = context {
  let sizes = current-sizes()
  let weights = current-weights()
  block(breakable: false, above: 0.1in, stroke: (top: light-rule),
    inset: (top: 0.05in))[
    #kicker-block([In Brief], sizes, weights)
    #v(0.06in)
    #for b in briefs {
      b
    }
  ]
}

// --- Section Label ---
#let section-label(heading) = context {
  let sizes = current-sizes()
  block(above: 0.15in, below: 0.08in, sticky: true, stroke: (top: dark-rule),
    inset: (top: 0.06in))[
    #text(sizes.section, weight: "bold", font: sans-font)[
      #upper(heading)
    ]
  ]
}

// --- Masthead ---
#let masthead(title, subtitle, date, editorial-note: none, debug-mode: false) = context {
  let sizes = current-sizes()
  block(below: 0.1in, stroke: (bottom: 3pt + text-color),
    inset: (bottom: 0.08in))[
    #align(center)[
      #text(sizes.masthead, weight: "bold", font: sans-font, tracking: 0.04em)[
        #set par(leading: 0.35em)
        #title
      ]
      #v(0.05in)
      #text(sizes.body, font: sans-font, fill: luma(68), tracking: 0.05em)[
        #upper(subtitle)
      ]
      #v(0.05in)
      #text(metadata-size, font: sans-font, fill: meta-color)[#date]
      #if debug-mode and editorial-note != none {
        v(0.1in)
        text(metadata-size, font: body-font, style: "italic", fill: luma(68))[#editorial-note]
      }
    ]
  ]
}

// --- Curation Summary ---
#let curation-summary(summary) = {
  if summary != none and summary != "" {
    block(above: 0.05in, below: 0.1in)[
      #text(8pt, font: sans-font, fill: luma(136))[
        #set par(leading: 0.45em)
        #summary
      ]
    ]
  }
}

// --- Colophon ---
#let colophon(title, subtitle, date) = {
  block(above: 0.3in, breakable: false)[
    #line(length: 100%, stroke: dark-rule)
    #v(0.1in)
    #align(center)[
      #text(12pt, weight: "bold", font: sans-font, fill: text-color)[#title]
      #v(0.03in)
      #text(8pt, font: sans-font, fill: meta-color)[#subtitle · #date]
      #v(0.03in)
      #text(7pt, font: sans-font, style: "italic", fill: light-meta)[Curated and composed automatically.]
    ]
  ]
}

// --- Row Composition ---
// Renders a row of columns as a grid
#let article-row(columns-data, ruled-indices: ()) = {
  let ncols = columns-data.len()
  if ncols == 0 { return }
  if ncols == 1 {
    columns-data.at(0)
    return
  }

  let col-widths = range(ncols).map(_ => 1fr)

  grid(
    columns: col-widths,
    column-gutter: column-gap,
    ..for (idx, col) in columns-data.enumerate() {
      if idx > 0 and idx in ruled-indices {
        (grid.vline(x: idx, stroke: rule-stroke),)
      }
      (col,)
    }
  )
}
