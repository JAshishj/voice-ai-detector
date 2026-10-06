# Design System: Voice AI Detector Console — "Signal Lab After Hours"

## 1. Visual Theme & Atmosphere
A forensic audio instrument interface: dark, clinical, confident. Density 5
(balanced), variance 6 (offset-asymmetric), motion 6 (fluid). The mood is a
well-calibrated lab bench after hours — quiet surfaces, one glowing signal
color, readouts set like test-equipment type. Never playful, never salesy.
The hero is the signal itself: a waveform that draws during analysis and
resolves into the verdict.

## 2. Color Palette & Roles
- **Void Canvas** (#131315) — primary background surface
- **Panel Surface** (#1C1C1F) — cards, analyzer rail, containers
- **Raised Surface** (#26262B) — active or elevated elements only
- **Ink** (#E8E8EA) — primary text (never pure white, never pure black)
- **Dim** (#8E8E93) — metadata, labels, secondary text
- **Hairline** (rgba(255,255,255,0.08)) — 1px structural dividers only
- **Oscilloscope Teal** (#2DD4BF) — THE single accent: CTAs, focus rings,
  waveform stroke, active states, analyzing pulse
- **Human Emerald** (#34D399) — functional verdict color for HUMAN only
- **Synthetic Rose** (#FB7185) — functional verdict color for AI_GENERATED only
Banned: #000000, purple/blue neon, gradients on text, outer glow shadows,
saturation above 80%.

## 3. Typography Rules
- **Display:** Space Grotesk, weight 700, letter-spacing -0.02em, fluid
  32–48px. Reserved for the verdict word and the console title.
- **Body:** Geist, 400/500, 15–16px, relaxed leading, 65ch max measure.
- **Data:** JetBrains Mono for ALL numbers — confidence scores, thresholds,
  timestamps, segment indexes, latencies, model metrics.
- Banned: Inter, generic serif faces, emoji in any UI copy.

## 4. Component Stylings
- **Buttons:** flat teal fill with near-black label, 12px radius, tactile
  -1px translate on press. Secondary: 1px hairline ghost, Ink text.
- **Cards:** panel fill, 20px radius, 1px hairline border. No drop shadows;
  depth comes from tonal layering only.
- **Inputs:** mono uppercase label above, panel fill, hairline border;
  focus transitions border to teal. Error/help text below in Rose or Dim.
- **Loaders:** skeletal shimmer blocks matching the result card's exact
  dimensions. No circular spinners.
- **Empty states:** composed — ghost waveform mark plus one action, never
  bare "No data" text.
- **Confidence bar:** 8px track on Raised Surface, verdict-color fill,
  mono readout right-aligned.
- **Segment strip:** mono-indexed chips showing per-window contribution for
  long clips.

## 5. Layout Principles
Asymmetric split: 7-column analyzer rail, 5-column verdict stage, 1280px
max-width, left-aligned with generous right margin. Collapse to a single
column below 768px with 16px margins. Hero section uses min-h-100dvh.
CSS grid first, no calc percentage hacks, no overlapping elements.

## 6. Motion & Interaction
Spring physics (stiffness 100, damping 20) on interactive elements.
Waveform draws itself while analyzing, then resolves into the verdict badge.
Result reveals use staggered cascades. Animate transform and opacity only.
Honor prefers-reduced-motion by rendering final states instantly.

## 7. Anti-Patterns (Banned)
Emojis; Inter; centered hero sections; 3-equal-card feature rows; neon
outer glows; copy clichés ("Seamless", "Elevate", "Unleash", "Next-Gen");
fake round statistics — use the real figures (96.3% held-out accuracy,
F1 0.964, AUC 0.992, EER 0.020, threshold 0.85); scroll-hint filler text;
overlapping text and imagery.
