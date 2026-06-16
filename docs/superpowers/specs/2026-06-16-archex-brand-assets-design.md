# archex brand assets design

## Objective

Create two separate visual assets in `assets/` derived from the approved reference direction:

1. A banner for README / social / project headers.
2. An icon-only logo for avatars, favicons, and compact brand use.

The approved direction is the trusted circuit halo concept: a verified core at the center, circuitry radiating outward, dark local-first palette, and a shared symbol grammar between banner and logo.

## Design decisions

### Banner

- Layout: left-aligned `archex` wordmark with tagline block, right-aligned verification core.
- Vertical balance: the text block is centered vertically within the banner band.
- Separation: the verification core sits far enough right that copy does not visually collide with the circuitry.
- Visual tone: dark teal field, cyan glow, off-white wordmark, low-noise circuitry.
- Messaging: emphasize trust, verification, provenance, and safe-to-act context rather than abstract “AI future” language.

### Logo

- Format: icon-only.
- Symbol: circular verified core with a checkmark, surrounded by short architectural/circuit spokes.
- Behavior: readable on dark backgrounds and at small sizes.
- Relationship to banner: the logo is a reduction of the banner’s right-side symbol, not a separate design language.

## Asset outputs

### Banner outputs

- Source HTML artifact in `assets/`
- PNG export in `assets/`
- SVG export in `assets/`

### Logo outputs

- Source HTML artifact in `assets/`
- PNG export in `assets/`
- SVG export in `assets/`

## Visual system

- Background: deep blue-teal / near-black.
- Accent: cyan / electric aqua glow.
- Text: off-white primary, muted blue-gray secondary.
- Geometry: precise circles, spokes, and nodes. No soft blob shapes.
- Density: controlled technical detail. Enough circuitry to imply code/architecture, not enough to become visual clutter.

## Non-goals

- No mascot.
- No photorealism.
- No generic purple gradient AI aesthetic.
- No separate wordmark asset in this request.
- No attempt to exactly reproduce the reference image; use it as style guidance only.

## Implementation plan

1. Build a banner HTML artifact sized for wide header use.
2. Build a logo HTML/SVG artifact sized for square icon use.
3. Export each to PNG.
4. Export each to SVG where the design remains fully vector-native.
5. Verify the generated files exist in `assets/` and visually match the approved concept.

## Acceptance criteria

- `assets/` contains separate banner and logo deliverables.
- Banner copy is vertically centered and does not overlap or press into the right-side symbol.
- Logo is icon-only and visually consistent with the banner symbol.
- Both assets feel like the same brand system.
- Outputs are suitable for direct project use without further editing.
