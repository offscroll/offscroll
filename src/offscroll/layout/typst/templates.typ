// OffScroll Typst Templates
// Article-type rendering functions for newspaper layout.
// These are imported by newspaper.typ and called by the Python
// markup builder via data substitution.

// --- Design Tokens (must match newspaper.typ) ---
#let body-font = "Source Serif 4"
#let sans-font = "Source Sans 3"
#let mono-font = "Source Code Pro"
#let body-font-size = 10pt
#let headline-feature = 28pt
#let headline-thread = 18pt
#let headline-standard = 14pt
#let metadata-size = 9pt
#let caption-size = 8pt
#let brief-size = 9pt
#let pull-quote-size = 14pt
#let column-gap = 0.25in
#let rule-stroke = 0.5pt + luma(221)
#let dark-rule = 2pt + luma(26)
#let light-rule = 0.5pt + luma(204)
#let text-color = luma(26)
#let meta-color = luma(102)
#let light-meta = luma(153)

// --- Pull Quote ---
#let pull-quote(text-content, attribution) = {
  block(breakable: false, above: 0.08in, below: 0.08in,
    align(center,
      block(width: 85%, stroke: (top: 1pt + text-color, bottom: 1pt + text-color),
        inset: (y: 0.08in))[
        #set text(pull-quote-size, font: body-font, style: "italic")
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

// --- Drop Cap ---
#let drop-cap(letter) = {
  box(
    baseline: 22pt,
    inset: (right: 0.06in, top: 0.02in),
    text(36pt, weight: "bold", font: sans-font, fill: text-color)[#letter]
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
  lead-text: "",
  lead-first-alpha: 0,
  body-paragraphs: (),
  inline-pq: none,
  inline-pq-idx: -1,
  edited-for-length: false,
) = {
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

    // Kicker
    #text(metadata-size, weight: "bold", font: sans-font, fill: meta-color)[
      #upper(kicker)
    ]
    #v(0.03in)

    // Title
    #if title != none {
      block(sticky: true)[
        #text(headline-feature, weight: "bold", font: sans-font)[
          #set par(leading: 0.4em)
          #title
        ]
      ]
      v(0.05in)
    }

    // Deck
    #if deck != none and deck != "" {
      text(body-font-size, font: body-font, style: "italic", fill: luma(68))[
        #set par(leading: 0.52em)
        #deck
      ]
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
      if lead-text != "" {
        let pre = lead-text.slice(0, lead-first-alpha)
        let cap-letter = lead-text.slice(lead-first-alpha, lead-first-alpha + 1)
        let rest = lead-text.slice(lead-first-alpha + 1)
        [#pre#drop-cap(cap-letter)#rest]
      }
    }
    #v(0.08in)

    // Body (2-column)
    #if body-paragraphs.len() > 0 {
      columns(2, gutter: column-gap)[
        #set par(justify: true)
        #set text(body-font-size, hyphenate: true)
        #for (idx, para) in body-paragraphs.enumerate() {
          if para.len() > 0 {
            [#para]
            v(0.05in)
          }
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
) = {
  block(breakable: word-count <= 200, below: 0.15in)[
    // Headline
    #if title != none {
      block(sticky: true)[
        #text(headline-standard, weight: "bold", font: sans-font)[
          #set par(leading: 0.4em)
          #title
        ]
      ]
      v(0.03in)
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
          set text(body-font-size, hyphenate: true)
          let extra-images = if images.len() > 1 { images.slice(1) } else { () }
          for (idx, para) in paragraphs.enumerate() {
            if para.len() > 0 {
              [#para]
              v(0.05in)
            }
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
) = {
  block(breakable: false, below: 0.15in)[
    // Headline
    #text(headline-thread, weight: "bold", font: sans-font)[
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
          #set text(body-font-size, hyphenate: true)
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
#let brief-group(briefs) = {
  block(breakable: false, above: 0.1in, stroke: (top: light-rule),
    inset: (top: 0.05in))[
    #text(metadata-size, weight: "bold", font: sans-font, fill: meta-color)[
      #upper[In Brief]
    ]
    #v(0.06in)
    #for b in briefs {
      b
    }
  ]
}

// --- Section Label ---
#let section-label(heading) = {
  block(above: 0.15in, below: 0.08in, stroke: (top: dark-rule),
    inset: (top: 0.06in))[
    #text(14pt, weight: "bold", font: sans-font)[
      #upper(heading)
    ]
  ]
}

// --- Masthead ---
#let masthead(title, subtitle, date, editorial-note: none, debug-mode: false) = {
  block(below: 0.1in, stroke: (bottom: 3pt + text-color),
    inset: (bottom: 0.08in))[
    #align(center)[
      #text(48pt, weight: "bold", font: sans-font, tracking: 0.04em)[
        #set par(leading: 0.35em)
        #title
      ]
      #v(0.05in)
      #text(body-font-size, font: sans-font, fill: luma(68), tracking: 0.05em)[
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
