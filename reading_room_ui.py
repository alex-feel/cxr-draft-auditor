"""
Reading Room UI for the CXR Draft Auditor Space: theme, CSS, and HTML builders.

Pure presentation - no torch, no gradio callbacks. ``app.py`` imports the CSS
string and the small HTML builders here and passes ``build_theme()``, ``CSS``,
and ``HEAD`` to ``demo.launch(...)`` (the Gradio 6 home for app-level theme/css
and injected ``<head>`` markup). Status colors intentionally match the overlay
renderer in ``cxr_auditor.render`` (green = supported, amber = missing from
draft, red = unsupported / urgent) so the chrome and the image annotations speak
one visual language.

Light and dark
--------------
The app ships TWO designs in one theme: a dark "Reading Room" (PACS-inspired) and
a light "Clinical Light" (warm paper). ``build_theme`` sets the light palette as
the base Gradio tokens and the dark palette as the ``_dark`` tokens, and the CSS
declares the ``--rr-*`` custom properties for light in ``:root`` and overrides
them for dark under ``body.dark`` (the class Gradio adds in dark mode). Because
every server-rendered chrome color (pills, chips, verdict cards, legend) is
emitted as ``var(--rr-*)`` rather than a baked hex, one served page recolors
correctly in either mode. A visitor switches with the header sun/moon toggle,
which reloads via the ``?__theme=`` URL parameter and remembers the choice in
``localStorage``; the ``HEAD`` script defaults a first-time visit to dark.

The module top level imports only the standard library, so every HTML builder
and the CSS string are importable and unit-testable in the pure-logic dev
environment where gradio is not installed. Only ``build_theme()`` touches
gradio, lazily via ``importlib``, because only the Space runtime (where the
platform provides gradio) ever calls it.
"""

from __future__ import annotations

import html
import importlib
import re
from typing import Any

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
# Two complete palettes keyed by the same names. DARK is Direction A
# ("Reading Room", PACS-inspired dark, the default look); LIGHT is Direction B
# ("Clinical Light", warm-paper). The keys feed both the Gradio theme tokens
# (``_theme_pairs``) and the CSS custom properties (``_palette_vars``).
DARK: dict[str, str] = {
    "page": "#0c1014",
    "panel": "#141a21",
    "panel_soft": "#10151b",
    "code_bg": "#0b0f13",
    "border": "rgba(255,255,255,0.09)",
    "ink": "#e9eef3",
    "muted": "#94a1ae",
    "faint": "#5d6a76",
    "accent": "#67c8c0",
    "accent_hover": "#7ed6cf",
    "accent_ink": "#06201e",
    "color_accent_soft": "rgba(103,200,192,0.15)",
    "green": "#3fce85",
    "amber": "#f0a52e",
    "red": "#ef6a5a",
    "green_bg": "rgba(46,204,113,0.13)",
    "amber_bg": "rgba(243,156,18,0.13)",
    "red_bg": "rgba(231,76,60,0.14)",
    "input_bg": "#06080a",
}

LIGHT: dict[str, str] = {
    "page": "#f2f3f1",
    "panel": "#ffffff",
    "panel_soft": "#f7f8f6",
    "code_bg": "#f3f4f2",
    "border": "#e4e6e2",
    "ink": "#1c2329",
    "muted": "#5d6973",
    "faint": "#97a1a9",
    "accent": "#11716b",
    "accent_hover": "#0e615c",
    "accent_ink": "#ffffff",
    "color_accent_soft": "rgba(17,113,107,0.12)",
    "green": "#177a45",
    "amber": "#9a6608",
    "red": "#bb3a2b",
    "green_bg": "rgba(23,122,69,0.10)",
    "amber_bg": "rgba(178,121,15,0.12)",
    "red_bg": "rgba(187,58,43,0.10)",
    "input_bg": "#f7f8f6",
}

# The evidence OUTPUT viewer (#overlay-out, where the scan and its colored boxes
# render) stays near-black in BOTH modes for maximum contrast, so this value is a
# constant. The upload INPUT zone (#xray-input), by contrast, follows the page
# theme (paper in light, near-black in dark) via the --rr-input-bg variable, to
# match the original Clinical Light design.
_VIEWER_BG = "#06080a"

# Typographic characters used in rendered output, built via chr() so this source
# file stays pure ASCII (the lint gate flags ambiguous non-ASCII literals in
# Python string values).
_MIDDLE_DOT = chr(0xB7)
_EM_DASH = chr(0x2014)
_LEFT_DOUBLE_QUOTE = chr(0x201C)
_RIGHT_DOUBLE_QUOTE = chr(0x201D)
_SUN = chr(0x2600)  # default toggle glyph (dark default -> offer "go light")


def _theme_pairs(palette: dict[str, str]) -> dict[str, str]:
    """Map a palette to the Gradio theme token names it drives.

    Args:
        palette: One of ``LIGHT`` / ``DARK``.

    Returns:
        A ``{gradio_token: value}`` dict ready for ``theme.set``.
    """
    return {
        "body_background_fill": palette["page"],
        "body_text_color": palette["ink"],
        "body_text_color_subdued": palette["muted"],
        "background_fill_primary": palette["panel"],
        "background_fill_secondary": palette["panel_soft"],
        "border_color_primary": palette["border"],
        "border_color_accent": palette["accent"],
        "block_background_fill": palette["panel"],
        "block_border_color": palette["border"],
        "block_label_background_fill": palette["panel"],
        "block_label_text_color": palette["muted"],
        "block_title_text_color": palette["ink"],
        "input_background_fill": palette["panel_soft"],
        "input_border_color": palette["border"],
        "input_placeholder_color": palette["faint"],
        "button_primary_background_fill": palette["accent"],
        "button_primary_background_fill_hover": palette["accent_hover"],
        "button_primary_text_color": palette["accent_ink"],
        "button_secondary_background_fill": "transparent",
        "button_secondary_border_color": palette["border"],
        "button_secondary_text_color": palette["muted"],
        "color_accent_soft": palette["color_accent_soft"],
        "link_text_color": palette["accent"],
        "table_border_color": palette["border"],
        "table_even_background_fill": palette["panel"],
        "table_odd_background_fill": palette["panel"],
        "panel_background_fill": palette["panel"],
        "panel_border_color": palette["border"],
    }


def build_theme() -> Any:
    """Build the adaptive Gradio theme (light base tokens, dark ``_dark`` tokens).

    Gradio is imported lazily via ``importlib`` so this module stays importable
    (and the pure HTML/CSS builders stay unit-testable) without gradio
    installed; only the Space app, which runs where the platform provides
    gradio, calls this. The light palette feeds the base tokens and the dark
    palette feeds the ``_dark`` tokens, so Gradio renders Clinical Light in light
    mode and Reading Room in dark mode; the ``HEAD`` script defaults a first
    visit to dark.

    Returns:
        A configured ``gradio.themes.Base`` instance (typed ``Any`` because
        gradio is not importable in the dev environment).
    """
    gr: Any = importlib.import_module("gradio")
    fonts: Any = importlib.import_module("gradio.themes.utils.fonts")
    theme: Any = gr.themes.Base(
        font=[fonts.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )
    light_pairs = _theme_pairs(LIGHT)
    dark_pairs = _theme_pairs(DARK)
    # Base tokens = light; ``_dark`` tokens = dark. Filter through hasattr so a
    # token rename in a future gradio version degrades gracefully instead of
    # crashing the Space at startup.
    candidate = {**light_pairs, **{f"{key}_dark": value for key, value in dark_pairs.items()}}
    return theme.set(**{key: value for key, value in candidate.items() if hasattr(theme, key)})


def _palette_vars(palette: dict[str, str]) -> str:
    """Render a palette as the ``--rr-*`` CSS custom-property declarations.

    Args:
        palette: One of ``LIGHT`` / ``DARK``.

    Returns:
        A semicolon-separated CSS declaration string (no surrounding braces).
    """
    return (
        f"--rr-page:{palette['page']};--rr-panel:{palette['panel']};--rr-panel-soft:{palette['panel_soft']};"
        f"--rr-code-bg:{palette['code_bg']};--rr-border:{palette['border']};--rr-ink:{palette['ink']};"
        f"--rr-muted:{palette['muted']};--rr-faint:{palette['faint']};--rr-accent:{palette['accent']};"
        f"--rr-accent-ink:{palette['accent_ink']};--rr-green:{palette['green']};--rr-green-bg:{palette['green_bg']};"
        f"--rr-amber:{palette['amber']};--rr-amber-bg:{palette['amber_bg']};--rr-red:{palette['red']};"
        f"--rr-red-bg:{palette['red_bg']};--rr-input-bg:{palette['input_bg']};"
    )


# ---------------------------------------------------------------------------
# CSS (light in :root, dark under body.dark; everything else reads the vars)
# ---------------------------------------------------------------------------
CSS: str = f"""
:root {{
  {_palette_vars(LIGHT)}
  --rr-radius: 14px;
}}
body.dark {{ {_palette_vars(DARK)} }}
.gradio-container {{
  max-width: 1340px !important;
  margin: 0 auto !important;
  background: var(--rr-page) !important;
}}
.gradio-container, .gradio-container * {{ font-family: 'IBM Plex Sans', ui-sans-serif, sans-serif; }}
code, pre, .rr-mono, .cm-editor * {{ font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }}

/* ---- header / ribbon / footer (rendered via gr.HTML) ---- */
#rr-header {{ display: flex; align-items: center; gap: 14px; padding: 14px 4px 12px; border-bottom: 1px solid var(--rr-border); }}
#rr-header .mark {{ width: 30px; height: 30px; border-radius: 8px; border: 2px solid var(--rr-accent); display: grid; place-items: center; flex: 0 0 auto; }}
#rr-header .mark i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--rr-accent); }}
#rr-header h1 {{ font-size: 19px; font-weight: 700; letter-spacing: -0.01em; color: var(--rr-ink); margin: 0; white-space: nowrap; }}
#rr-header .tag {{ font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--rr-muted); }}
#rr-header .spacer {{ flex: 1; }}
#rr-header .chip {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 4px 10px; border-radius: 99px; border: 1px solid var(--rr-border); color: var(--rr-muted); background: var(--rr-panel-soft); white-space: nowrap; }}
#rr-header .chip.warn {{ color: var(--rr-amber); background: var(--rr-amber-bg); border-color: var(--rr-amber-bg); font-weight: 600; }}
#rr-header .chip.model {{ color: var(--rr-ink); border-color: var(--rr-accent); }}
#rr-header .chip.model b {{ color: var(--rr-accent); font-weight: 700; }}
#rr-header a.chip.model {{ text-decoration: none; cursor: pointer; transition: background .15s, color .15s, border-color .15s; }}
#rr-header a.chip.model:hover {{ background: var(--rr-accent); border-color: var(--rr-accent); color: var(--rr-accent-ink); }}
#rr-header a.chip.model:hover b {{ color: var(--rr-accent-ink); }}
#rr-header a.chip.blog {{ text-decoration: none; cursor: pointer; color: var(--rr-accent-ink); background: var(--rr-accent); border-color: var(--rr-accent); font-weight: 600; transition: opacity .15s; }}
#rr-header a.chip.blog:hover {{ opacity: 0.85; }}
#rr-header .rr-toggle {{ font-family: 'IBM Plex Mono', monospace; font-size: 14px; line-height: 1; padding: 5px 10px; border-radius: 99px; border: 1px solid var(--rr-border); background: var(--rr-panel-soft); color: var(--rr-muted); cursor: pointer; flex: 0 0 auto; }}
#rr-header .rr-toggle:hover {{ color: var(--rr-ink); border-color: var(--rr-accent); }}
#rr-ribbon {{ display: flex; gap: 10px; align-items: center; padding: 9px 4px; border-bottom: 1px solid var(--rr-border); margin-bottom: 10px; }}
#rr-ribbon i {{ width: 6px; height: 6px; border-radius: 50%; background: var(--rr-amber); flex: 0 0 auto; }}
#rr-ribbon p {{ font-size: 12.5px; color: var(--rr-muted); margin: 0; }}
#rr-ribbon strong {{ color: var(--rr-ink); font-weight: 600; }}
#rr-footer {{ display: flex; gap: 16px; padding: 14px 4px; border-top: 1px solid var(--rr-border); margin-top: 22px; }}
#rr-footer span {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--rr-faint); }}
#rr-footer .spacer {{ flex: 1; }}
#rr-footer a {{ color: var(--rr-muted); text-decoration: underline; text-underline-offset: 2px; transition: color .15s; }}
#rr-footer a:hover {{ color: var(--rr-accent); }}
#rr-footer a.rr-hackathon {{ color: var(--rr-accent); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}

/* ---- panels ---- */
.rr-panel {{ background: var(--rr-panel) !important; border: 1px solid var(--rr-border) !important; border-radius: var(--rr-radius) !important; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
.rr-panel > * {{ border: none !important; }}
/* Gradio 6 wraps each gr.Group child stack in a .styler div that has its own
   overflow:hidden and a 0 radius, sized to exactly fill the rounded .rr-panel.
   That square clip rectangle coincides with the panel edges and defeats the
   panel's rounded overflow clip at the bottom, so the near-black image block
   below paints square bottom corners. Round this wrapper's bottom corners to
   the panel radius so its clip shape follows the panel; the top stays square
   because the panel itself provides the rounded top and the card-head meets it. */
.rr-panel > .styler {{ border-bottom-left-radius: var(--rr-radius); border-bottom-right-radius: var(--rr-radius); }}
.rr-head {{ display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--rr-border); }}
.rr-head .step {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; color: var(--rr-accent-ink); background: var(--rr-accent); border-radius: 6px; padding: 2px 7px; }}
.rr-head .title {{ font-weight: 600; font-size: 14.5px; color: var(--rr-ink); white-space: nowrap; }}
.rr-head .spacer {{ flex: 1; }}
.rr-head .meta {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--rr-faint); }}

/* legend chips */
.rr-legend {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.rr-legend span {{ display: inline-flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--rr-muted); }}
.rr-legend i {{ width: 10px; height: 10px; border-radius: 2px; box-sizing: border-box; }}

/* ---- inputs ---- */
/* Both the upload input and the evidence output are the bottom-most child of
   their panel: keep their tops square (to meet the card-head divider flush) and
   round their bottom corners to the panel radius so the background follows the
   panel's rounded bottom instead of poking square corners past it. The upload
   INPUT zone follows the page theme (paper in light, near-black in dark) via
   --rr-input-bg, matching the Clinical Light design; the evidence OUTPUT viewer
   stays near-black in both modes so the scan and its boxes keep maximum contrast. */
#xray-input {{
  background: var(--rr-input-bg) !important;
  border-top-left-radius: 0 !important; border-top-right-radius: 0 !important;
  border-bottom-left-radius: var(--rr-radius) !important; border-bottom-right-radius: var(--rr-radius) !important;
}}
#overlay-out {{
  background: {_VIEWER_BG} !important;
  border-top-left-radius: 0 !important; border-top-right-radius: 0 !important;
  border-bottom-left-radius: var(--rr-radius) !important; border-bottom-right-radius: var(--rr-radius) !important;
}}
#xray-input .upload-container, #xray-input .wrap {{ color: var(--rr-muted) !important; }}
#draft-input textarea {{
  background: var(--rr-panel-soft) !important; border: 1px solid var(--rr-border) !important;
  border-radius: 12px !important; color: var(--rr-ink) !important; font-size: 13.5px !important; line-height: 1.55 !important;
}}
#run-btn {{ font-weight: 600 !important; font-size: 15px !important; border-radius: 12px !important; padding: 13px 0 !important; box-shadow: 0 6px 16px rgba(0,0,0,0.14) !important; }}
#clear-btn {{ border-radius: 12px !important; font-size: 15px !important; padding: 13px 0 !important; }}

/* ---- verdict / table / json ---- */
.rr-chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
.rr-chips b {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; border-radius: 99px; padding: 4px 10px; }}
.rr-item {{ display: flex; gap: 12px; padding: 11px 13px; border-radius: 10px; align-items: flex-start; margin-bottom: 8px; }}
.rr-item i {{ width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex: 0 0 auto; }}
.rr-item p {{ font-size: 12.8px; line-height: 1.5; color: var(--rr-ink); margin: 0; }}
.rr-item code {{ font-size: 11.5px; }}
.rr-empty {{ font-size: 12.5px; color: var(--rr-faint); padding: 6px 2px; }}

table.rr-table {{ width: 100%; border-collapse: collapse; }}
table.rr-table th {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--rr-faint); text-align: left; padding: 8px 14px; border-bottom: 1px solid var(--rr-border); }}
table.rr-table td {{ font-size: 13px; color: var(--rr-ink); padding: 10px 14px; border-bottom: 1px solid var(--rr-border); }}
table.rr-table tr:last-child td {{ border-bottom: none; }}
table.rr-table td.dim {{ color: var(--rr-muted); }}
.rr-pill {{ display: inline-flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; font-weight: 600; border-radius: 99px; padding: 3px 9px; }}
.rr-pill i {{ width: 6px; height: 6px; border-radius: 50%; }}

#json-out {{ border-radius: 10px; overflow: hidden; }}
#json-out .cm-editor, #json-out pre {{ background: var(--rr-code-bg) !important; font-size: 11.5px !important; }}

/* gradio chrome cleanup */
.rr-pad {{ padding: 12px 16px; }}
footer {{ opacity: 0.55; }}
.rr-panel .block {{ background: transparent !important; }}

/* ---- mobile (<=700px): stack the two columns and keep every panel within the
   viewport, so a phone never scrolls horizontally. Scoped entirely to this media
   query, so the desktop and tablet layout above is untouched. The custom header,
   footer, and card-heads are non-wrapping flex rows whose nowrap children would
   otherwise pin the container wider than the viewport and defeat column stacking;
   here they wrap and their nowrap text relaxes, and the two evidence columns are
   forced onto full-width flex lines with no min-width floor so they stack.
   Critically, the container is capped at the viewport width and clips horizontal
   overflow, so a child with a wide intrinsic size - the raw-JSON code editor's
   long lines or a wide findings table - scrolls inside its own box instead of
   expanding the whole page out past the screen. The evidence image is made
   width-driven (height auto) so it fills the column rather than forcing its tall
   fixed-height aspect width. The ribbon is deliberately NOT wrapped: its dot and
   text share one flex line, so wrapping it would strand the dot alone on a line
   above the text. ---- */
@media (max-width: 700px) {{
  .gradio-container {{ max-width: 100vw !important; overflow-x: hidden !important; }}
  #rr-header {{ flex-wrap: wrap; }}
  #rr-header h1 {{ white-space: normal; }}
  #rr-header .chip {{ white-space: normal; }}
  #rr-header .spacer {{ flex-basis: 100%; }}
  #rr-footer {{ flex-wrap: wrap; }}
  #rr-footer .spacer {{ flex-basis: 100%; }}
  .rr-head {{ flex-wrap: wrap; }}
  .rr-head .title {{ white-space: normal; }}
  .rr-row {{ flex-wrap: wrap !important; }}
  .rr-col {{ flex: 1 1 100% !important; min-width: 0 !important; }}
  #overlay-out {{ height: auto !important; min-height: 220px !important; }}
  .rr-tablewrap {{ overflow-x: auto !important; }}
  #json-out .cm-scroller {{ overflow-x: auto !important; }}
}}
"""


# ---------------------------------------------------------------------------
# HEAD: early script that defaults a first visit to dark and wires the toggle
# ---------------------------------------------------------------------------
# Passed to ``demo.launch(head=...)``. It runs during head parse, before the app
# renders, so the default-dark redirect happens without a light-mode flash. The
# toggle (a header button with id ``rr-theme-toggle``) reloads with the flipped
# ``?__theme=`` value, which Gradio honors natively (dark/light/system), and the
# choice persists in localStorage. Gradio applies dark mode as ``body.dark``, so
# the CSS overrides above take effect for the chosen mode. Written as a raw
# string so the ``\\uXXXX`` glyph escapes stay literal for JavaScript and this
# Python source remains pure ASCII. Degrades gracefully: any failure is swallowed
# and the app still works (just without the in-app switch).
HEAD: str = r"""<script>
(function () {
  var KEY = 'cxr-theme';
  try {
    var url = new URL(window.location.href);
    var param = url.searchParams.get('__theme');
    if (param !== 'light' && param !== 'dark') {
      var stored = null;
      try { stored = window.localStorage.getItem(KEY); } catch (e) {}
      url.searchParams.set('__theme', stored === 'light' ? 'light' : 'dark');
      window.location.replace(url.toString());
      return;
    }
    var mode = param;
    try { window.localStorage.setItem(KEY, mode); } catch (e) {}
    var wire = function () {
      var btn = document.getElementById('rr-theme-toggle');
      if (!btn) { return false; }
      var toLight = mode === 'dark';
      btn.innerHTML = toLight ? String.fromCharCode(0x2600, 0xFE0E) : String.fromCharCode(0x263D);
      btn.title = toLight ? 'Switch to light theme' : 'Switch to dark theme';
      btn.setAttribute('aria-label', btn.title);
      if (!btn.dataset.wired) {
        btn.dataset.wired = '1';
        btn.addEventListener('click', function () {
          var next = mode === 'dark' ? 'light' : 'dark';
          try { window.localStorage.setItem(KEY, next); } catch (e) {}
          var u2 = new URL(window.location.href);
          u2.searchParams.set('__theme', next);
          window.location.assign(u2.toString());
        });
      }
      return true;
    };
    if (!wire()) {
      var tries = 0;
      var iv = setInterval(function () { if (wire() || ++tries > 100) { clearInterval(iv); } }, 100);
    }
  } catch (e) {}
})();
</script>
<script>
(function () {
  function align() {
    try {
      var overlay = document.getElementById('overlay-out');
      var draftEl = document.getElementById('draft-input');
      if (!overlay || !draftEl) { return; }
      var draftPanel = draftEl.closest('.rr-panel') || draftEl;
      var o = overlay.getBoundingClientRect();
      var d = draftPanel.getBoundingClientRect();
      if (o.left >= d.right - 4) {
        var h = Math.round(d.bottom - o.top);
        if (h >= 280) {
          if (overlay.style.height !== h + 'px') { overlay.style.height = h + 'px'; }
          return;
        }
      }
      if (overlay.style.height !== '520px') { overlay.style.height = '520px'; }
    } catch (e) {}
  }
  var raf = null;
  function schedule() { if (raf) { cancelAnimationFrame(raf); } raf = requestAnimationFrame(align); }
  function setup() {
    var overlay = document.getElementById('overlay-out');
    var draftEl = document.getElementById('draft-input');
    if (!overlay || !draftEl) { return false; }
    var draftPanel = draftEl.closest('.rr-panel') || draftEl;
    if (window.ResizeObserver) {
      var ro = new ResizeObserver(schedule);
      ro.observe(draftPanel);
      ro.observe(document.body);
    }
    if (window.MutationObserver) {
      var mo = new MutationObserver(schedule);
      mo.observe(overlay, { childList: true, subtree: true });
    }
    window.addEventListener('resize', schedule);
    schedule();
    return true;
  }
  if (!setup()) {
    var tries = 0;
    var iv = setInterval(function () { if (setup() || ++tries > 100) { clearInterval(iv); } }, 100);
  }
})();
</script>"""


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------


def header_html(
    grounding_model: str,
    draft_model: str,
    grounding_url: str,
    draft_url: str,
    blog_url: str,
) -> str:
    """Build the app header with the wordmark, toggle, blog link, chips, and badge.

    A filled accent BLOG POST chip (left of the model chips) links to the published
    write-up. Two accented model chips make the two-model pipeline explicit and
    prominent: the image-grounding model (MedGemma) and the draft parser (Nemotron).
    Each chip is a whole-chip link to its Hugging Face page, opened in a new tab.

    Args:
        grounding_model: Short name of the image-grounding model (escaped here).
        draft_model: Short name of the draft-parsing model (escaped here).
        grounding_url: Hub page the grounding chip links to (escaped here).
        draft_url: Hub page the draft chip links to (escaped here).
        blog_url: Target of the BLOG POST chip, the published write-up (escaped here).

    Returns:
        The header HTML. The toggle button (``id="rr-theme-toggle"``) is wired by
        the ``HEAD`` script; its glyph defaults to a sun (dark default offers
        "go light") until the script sets it for the active mode.
    """
    return f"""
<div id="rr-header">
  <div class="mark"><i></i></div>
  <h1>CXR Draft Auditor</h1>
  <span class="tag">second-look QA for draft impressions</span>
  <a class="chip blog" href="{html.escape(blog_url)}" target="_blank" rel="noopener noreferrer">BLOG POST</a>
  <span class="spacer"></span>
  <button id="rr-theme-toggle" class="rr-toggle" type="button" aria-label="Switch theme">{_SUN}</button>
  <a class="chip model" href="{html.escape(grounding_url)}" target="_blank" rel="noopener noreferrer"><b>grounding</b> {html.escape(grounding_model)}</a>
  <a class="chip model" href="{html.escape(draft_url)}" target="_blank" rel="noopener noreferrer"><b>draft</b> {html.escape(draft_model)}</a>
  <span class="chip warn">RESEARCH PREVIEW</span>
</div>"""


def ribbon_html() -> str:
    """Build the permanent research-use disclaimer ribbon under the header.

    Returns:
        The ribbon HTML.
    """
    return """
<div id="rr-ribbon"><i></i>
  <p>Research / educational QA only. <strong>Not a medical device, not a diagnosis, not for clinical use.</strong>
  Always confirm with a qualified radiologist.</p>
</div>"""


def footer_html(grounding_model_id: str, draft_model_id: str) -> str:
    """Build the footer line with both model ids and serving details.

    Args:
        grounding_model_id: Full Hugging Face id of the grounding model (escaped).
        draft_model_id: Full Hugging Face id of the draft-parser model (escaped).

    Returns:
        The footer HTML.
    """
    return f"""
<div id="rr-footer">
  <span>grounding <a href="https://huggingface.co/{html.escape(grounding_model_id)}" target="_blank" rel="noopener noreferrer">{html.escape(grounding_model_id)}</a> {_MIDDLE_DOT} draft <a href="https://huggingface.co/{html.escape(draft_model_id)}" target="_blank" rel="noopener noreferrer">{html.escape(draft_model_id)}</a> {_MIDDLE_DOT} ZeroGPU</span>
  <span class="spacer"></span>
  <span><a class="rr-hackathon" href="https://huggingface.co/build-small-hackathon" target="_blank" rel="noopener noreferrer">Build Small Hackathon</a> {_MIDDLE_DOT} 2026</span>
</div>"""


def card_head(step: str | None, title: str, meta: str = "", legend: bool = False) -> str:
    """Build a panel head row with an optional step badge and right-side slot.

    Args:
        step: Step badge text (for example ``"1"``), or ``None`` for no badge.
        title: The panel title.
        meta: Small right-aligned mono text; ignored when ``legend`` is set.
        legend: When ``True``, render the evidence color legend on the right.

    Returns:
        The panel head HTML.
    """
    step_html = f'<span class="step">{html.escape(step)}</span>' if step else ""
    right = legend_html() if legend else (f'<span class="meta">{html.escape(meta)}</span>' if meta else "")
    return f"""
<div class="rr-head">{step_html}<span class="title">{html.escape(title)}</span>
  <span class="spacer"></span>{right}
</div>"""


def legend_html() -> str:
    """Build the evidence color legend for the image overlay.

    Lists only the categories the overlay actually draws as boxes: supported
    (green), missing from draft (amber), and urgent (red, doubled border). An
    unsupported claim is asserted in the draft but absent from the image, so it
    has no box to draw and is deliberately omitted here; it surfaces in the audit
    verdict and the findings table instead (see ``cxr_auditor.render``). The
    swatches read the status CSS variables so the legend recolors with the active
    light/dark mode, exactly like the pills and verdict cards.

    Returns:
        The legend HTML.
    """
    return """
<div class="rr-legend">
  <span><i style="border:2px solid var(--rr-green)"></i>supported</span>
  <span><i style="border:2px solid var(--rr-amber)"></i>missing from draft</span>
  <span><i style="border:3px double var(--rr-red)"></i>urgent</span>
</div>"""


# Status name -> (color var, background-tint var). Keys are the four audit
# statuses the chrome distinguishes; unsupported and urgent share the alert red.
# Values are CSS custom properties (not baked hex) so one server-rendered panel
# recolors correctly in whichever light/dark mode the visitor is viewing.
_STATUS: dict[str, tuple[str, str]] = {
    "supported": ("var(--rr-green)", "var(--rr-green-bg)"),
    "missing": ("var(--rr-amber)", "var(--rr-amber-bg)"),
    "urgent": ("var(--rr-red)", "var(--rr-red-bg)"),
    "unsupported": ("var(--rr-red)", "var(--rr-red-bg)"),
}

# Audit-flag phrases exactly as emitted by ``cxr_auditor.render.findings_table_rows``.
_FLAG_TO_STATUS: dict[str, str] = {
    "Supported by image": "supported",
    "Missing from draft": "missing",
    "Urgent - review": "urgent",
    "Unsupported claim": "unsupported",
}

# Inline Markdown conversions for ``message_html``: the serving-error and status
# messages from ``cxr_auditor.inference`` use **bold** and `code` spans.
_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
_CODE_PATTERN = re.compile(r"`([^`]+)`")


def _markdown_to_inline_html(text: str) -> str:
    """Escape a Markdown-ish message and convert bold/code spans to HTML.

    Args:
        text: A plain message that may carry ``**bold**`` and ```code``` spans.

    Returns:
        HTML-safe text with ``<strong>`` and ``<code>`` spans.
    """
    escaped = html.escape(text)
    escaped = _BOLD_PATTERN.sub(r"<strong>\1</strong>", escaped)
    return _CODE_PATTERN.sub(r"<code>\1</code>", escaped)


def _pill(flag: str) -> str:
    """Render an audit-flag phrase as a colored status pill.

    Args:
        flag: The audit-flag phrase from ``findings_table_rows`` (may be empty).

    Returns:
        A pill ``<span>`` for a known phrase, or a faint dash placeholder.
    """
    status = _FLAG_TO_STATUS.get(flag)
    if status is None:
        return f'<span style="color: var(--rr-faint)">{_EM_DASH}</span>'
    color, bg = _STATUS[status]
    return (
        f'<span class="rr-pill" style="color:{color};background:{bg}">'
        f'<i style="background:{color}"></i>{html.escape(flag)}</span>'
    )


def table_html(rows: list[list[str]]) -> str:
    """Build the findings table with status pills.

    Args:
        rows: ``[finding, source, status, audit_flag]`` rows, exactly as
            returned by ``cxr_auditor.render.findings_table_rows``.

    Returns:
        A styled ``<table>``, or a placeholder line when there are no rows.
    """
    if not rows:
        return '<div class="rr-empty" style="padding:14px 16px">Run an audit to see the structured findings.</div>'
    body = "".join(
        f"<tr><td style='font-weight:550'>{html.escape(row[0])}</td>"
        f"<td class='dim'>{html.escape(row[1])}</td>"
        f"<td class='dim'>{html.escape(row[2])}</td>"
        f"<td>{_pill(row[3])}</td></tr>"
        for row in rows
    )
    return (
        "<table class='rr-table'><thead><tr>"
        "<th>Finding</th><th>Source</th><th>Status</th><th>Audit flag</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _chip(count: int, word: str, status: str) -> str:
    """Render one verdict summary chip (for example ``1 URGENT``).

    Args:
        count: Number of flagged items in the category.
        word: The category word, uppercased by the caller.
        status: A ``_STATUS`` key driving the chip tint.

    Returns:
        The chip HTML.
    """
    color, bg = _STATUS[status]
    return f'<b style="color:{color};background:{bg}">{count} {word}</b>'


def _item(status: str, head: str, body: str) -> str:
    """Render one tinted verdict card.

    Args:
        status: A ``_STATUS`` key driving the card tint and dot color.
        head: The bolded lead (a finding display name or a short label).
        body: The plain-English explanation.

    Returns:
        The card HTML (head and body are escaped here).
    """
    color, bg = _STATUS[status]
    return (
        f'<div class="rr-item" style="background:{bg}"><i style="background:{color}"></i>'
        f"<p><strong>{html.escape(head)}</strong> &mdash; {html.escape(body)}</p></div>"
    )


def verdict_html(
    urgent: list[str],
    missing: list[str],
    unsupported: list[tuple[str, str | None]],
    draft_parse_note: str | None,
) -> str:
    """Build the audit verdict panel: summary chips plus tinted per-flag cards.

    Args:
        urgent: Display names of urgent findings.
        missing: Display names of findings missing from the draft.
        unsupported: ``(display_name, draft_span)`` pairs for unsupported claims.
        draft_parse_note: Degradation note when the draft could not be parsed,
            or ``None``.

    Returns:
        The verdict HTML. When nothing is flagged, a green agreement card.
    """
    parts: list[str] = []
    if draft_parse_note:
        parts.append(_item("missing", "Draft not analyzed", f"{draft_parse_note} Re-check the draft text manually."))
    chips: list[str] = []
    if urgent:
        chips.append(_chip(len(urgent), "URGENT", "urgent"))
    if missing:
        chips.append(_chip(len(missing), "MISSING", "missing"))
    if unsupported:
        chips.append(_chip(len(unsupported), "UNSUPPORTED", "unsupported"))
    if chips:
        parts.append(f'<div class="rr-chips">{"".join(chips)}</div>')
    for name in urgent:
        parts.append(
            _item("urgent", name, f"Visible on the image and on the can't-miss list {_EM_DASH} have a radiologist review it.")
        )
    for name in missing:
        parts.append(_item("missing", name, "Appears on the image but the draft does not report it."))
    for name, span in unsupported:
        detail = (
            f"Stated in the draft ({_LEFT_DOUBLE_QUOTE}{span}{_RIGHT_DOUBLE_QUOTE}) but the image does not support it."
            if span
            else "Stated in the draft but the image does not support it."
        )
        parts.append(_item("unsupported", name, detail))
    if not (urgent or missing or unsupported):
        parts.append(
            _item(
                "supported",
                "No discrepancies found",
                "The draft and the image findings agree across the finding set the tool checks.",
            )
        )
    return "".join(parts)


def verdict_placeholder() -> str:
    """Build the verdict panel placeholder shown before any audit has run.

    Returns:
        The placeholder HTML.
    """
    return '<div class="rr-empty">Upload an X-ray (and optionally a draft impression), then press <strong>Run audit</strong>.</div>'


def message_html(markdown_message: str, status: str = "missing") -> str:
    """Render a standalone tinted message card in the verdict panel.

    Used for the model-configuration notice (amber) and for categorized audit
    failures (red), so degraded states share the verdict cards' visual language.

    Args:
        markdown_message: A message that may carry ``**bold**`` and ```code```
            spans (for example from ``categorize_serving_error``).
        status: A ``_STATUS`` key driving the card tint.

    Returns:
        The message card HTML.

    Raises:
        KeyError: If ``status`` is not one of the known status names.
    """
    color, bg = _STATUS[status]
    return (
        f'<div class="rr-item" style="background:{bg}"><i style="background:{color}"></i>'
        f"<p>{_markdown_to_inline_html(markdown_message)}</p></div>"
    )


__all__ = [
    "CSS",
    "DARK",
    "HEAD",
    "LIGHT",
    "build_theme",
    "card_head",
    "footer_html",
    "header_html",
    "legend_html",
    "message_html",
    "ribbon_html",
    "table_html",
    "verdict_html",
    "verdict_placeholder",
]
