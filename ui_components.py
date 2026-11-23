"""
ui_components.py - Readable Styling & Interactive Formula Explorer
=================================================================
Provides clean CSS typography and a self-contained HTML/JS component
where hovering or clicking any variable or constant highlights every
matching occurrence across the symbolic equation, live substituted rows,
and explanation cards. Works 100% offline.
"""

import html
import json
from typing import Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components


SOURCE_COLORS: Dict[str, Dict[str, str]] = {
    "Input": {"c": "#2563eb", "bg": "#dbeafe"},
    "HMM": {"c": "#9333ea", "bg": "#f3e8ff"},
    "MDP": {"c": "#ea580c", "bg": "#ffedd5"},
    "Route data": {"c": "#0d9488", "bg": "#ccfbf1"},
    "Constant": {"c": "#475569", "bg": "#e2e8f0"},
}


def inject_css() -> None:
    """Injects clean, high-readability custom styles into the Streamlit app."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1240px; }
        h1 { font-size: 2.0rem !important; letter-spacing: -0.02em; margin-bottom: 0.2rem !important; }
        h2, h3 { letter-spacing: -0.01em; margin-top: 0.8rem !important; }
        p, li { font-size: 1.0rem; line-height: 1.55; }
        div[data-testid="stMetric"] {
            background: rgba(148, 163, 184, 0.10);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            padding: 12px 14px;
        }
        div[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700; }
        div[data-testid="stMetricLabel"] { font-size: 0.86rem !important; }
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sym(key: str, text: str) -> str:
    """Wraps a symbol or substituted value so hovering highlights all matching keys."""
    safe_key = html.escape(key, quote=True)
    return f'<span class="v" data-k="{safe_key}">{text}</span>'


def interactive_formula(
    title: str,
    formula_html: str,
    variables: List[Dict[str, str]],
    rows_html: Optional[str] = None,
    footnote: str = "",
    height: int = 440,
) -> None:
    """
    Renders a self-contained, offline interactive formula card using st.components.v1.html.
    Hovering or clicking any symbol, live number, or legend card highlights all matching
    elements and explains what the variable/constant means and how it affects the result.
    """
    per_key_css = []
    cards_html = []
    var_map: Dict[str, Dict[str, str]] = {}

    for var in variables:
        k = var["key"]
        src = var.get("source", "Constant")
        palette = SOURCE_COLORS.get(src, SOURCE_COLORS["Constant"])
        c, bg = palette["c"], palette["bg"]
        safe_k = html.escape(k, quote=True)
        per_key_css.append(f'[data-k="{safe_k}"] {{ --c: {c}; --bg: {bg}; }}')

        var_map[k] = {
            "name": var["name"],
            "value": var["value"],
            "source": src,
            "effect": var["effect"],
            "color": c,
            "bg": bg,
        }

        cards_html.append(
            f"""
            <div class="card" data-k="{safe_k}" tabindex="0">
              <div class="card-top">
                <span class="card-sym">{var['symbol']}</span>
                <span class="src-badge" style="color:{c};background:{bg};">{html.escape(src)}</span>
              </div>
              <div class="card-name">{html.escape(var['name'])}</div>
              <div class="card-val">{html.escape(var['value'])}</div>
              <div class="card-eff">{html.escape(var['effect'])}</div>
            </div>
            """
        )

    source_pills = "".join(
        f'<span class="pill" style="color:{p["c"]};background:{p["bg"]};">{html.escape(name)}</span>'
        for name, p in SOURCE_COLORS.items()
    )

    rows_section = ""
    if rows_html:
        rows_section = f"""
        <div class="sub-block">
          <div class="sub-title">With your current numbers:</div>
          <div class="rows-wrap">{rows_html}</div>
        </div>
        """

    foot_section = f'<div class="footnote">{footnote}</div>' if footnote else ""
    uid = f"fx_{abs(hash(title)) % 1000000}"

    page_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ margin: 0; padding: 2px; background: transparent; color: #0f172a; }}
  .box {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
  }}
  .hdr {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .title {{ font-size: 16px; font-weight: 700; color: #0f172a; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .pill {{
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 999px;
  }}
  .formula-bar {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 20px;
    line-height: 1.65;
    font-weight: 600;
    color: #1e293b;
    overflow-x: auto;
  }}
  .v {{
    display: inline-block;
    padding: 1px 6px;
    margin: 0 1px;
    border-radius: 6px;
    color: var(--c, #334155);
    background: var(--bg, #f1f5f9);
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.14s ease;
  }}
  .v.hl {{
    background: var(--c, #2563eb) !important;
    color: #ffffff !important;
    box-shadow: 0 0 0 3px var(--bg, #dbeafe), 0 3px 10px rgba(15, 23, 42, 0.18);
    transform: translateY(-1px);
  }}
  .explain {{
    margin-top: 10px;
    min-height: 46px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 13px;
    border-radius: 10px;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    font-size: 13.5px;
    color: #334155;
    transition: background 0.15s ease, border-color 0.15s ease;
  }}
  .explain.active {{
    background: #eff6ff;
    border-color: #93c5fd;
    color: #0f172a;
  }}
  .sub-block {{ margin-top: 12px; }}
  .sub-title {{ font-size: 12.5px; font-weight: 700; color: #475569; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.03em; }}
  .rows-wrap {{
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 13.5px;
  }}
  .calc-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 11px;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
  }}
  .calc-row.winner {{
    background: #ecfdf5;
    border-color: #6ee7b7;
    font-weight: 600;
  }}
  .row-name {{ font-weight: 600; min-width: 195px; color: #0f172a; }}
  .row-expr {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; color: #1e293b; }}
  .row-total {{ font-weight: 700; color: #0f172a; }}
  .cards-grid {{
    margin-top: 14px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 8px;
  }}
  .card {{
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 10px;
    padding: 9px 11px;
    cursor: pointer;
    transition: all 0.14s ease;
  }}
  .card.hl {{
    border-color: var(--c, #2563eb);
    background: var(--bg, #eff6ff);
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.10);
    transform: translateY(-2px);
  }}
  .card-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }}
  .card-sym {{ font-size: 15px; font-weight: 800; color: var(--c, #0f172a); }}
  .src-badge {{ font-size: 10.5px; font-weight: 700; padding: 2px 6px; border-radius: 999px; }}
  .card-name {{ font-size: 12.5px; font-weight: 700; color: #0f172a; }}
  .card-val {{ font-size: 12.5px; font-weight: 600; color: #334155; margin-top: 1px; }}
  .card-eff {{ font-size: 11.5px; color: #475569; margin-top: 3px; line-height: 1.35; }}
  .footnote {{ margin-top: 10px; font-size: 12px; color: #64748b; }}
  {''.join(per_key_css)}
</style>
</head>
<body>
<div class="box" id="{uid}">
  <div class="hdr">
    <div class="title">{html.escape(title)}</div>
    <div class="pills">{source_pills}</div>
  </div>
  <div class="formula-bar">{formula_html}</div>
  <div class="explain" id="{uid}_exp">
    <span>💡 Hover or click any colored symbol, number, or card below to see what it means and how it affects the result.</span>
  </div>
  {rows_section}
  <div class="cards-grid">{''.join(cards_html)}</div>
  {foot_section}
</div>
<script>
(function() {{
  const root = document.getElementById({json.dumps(uid)});
  const expBox = document.getElementById({json.dumps(uid + "_exp")});
  const VARS = {json.dumps(var_map)};
  const DEFAULT_MSG = "💡 Hover or click any colored symbol, number, or card below to see what it means and how it affects the result.";
  let pinnedKey = null;

  function applyHighlight(key) {{
    root.querySelectorAll('[data-k]').forEach(el => {{
      if (key && el.getAttribute('data-k') === key) {{
        el.classList.add('hl');
      }} else {{
        el.classList.remove('hl');
      }}
    }});

    if (key && VARS[key]) {{
      const info = VARS[key];
      const pinNote = pinnedKey === key ? ' <span style="color:#64748b;font-size:12px;">(Pinned — click again to unpin)</span>' : '';
      expBox.classList.add('active');
      expBox.innerHTML =
        '<span class="src-badge" style="color:' + info.color + ';background:' + info.bg + ';">' + info.source + '</span>' +
        '<div><strong>' + info.name + ' = ' + info.value + '</strong> — ' + info.effect + pinNote + '</div>';
    }} else {{
      expBox.classList.remove('active');
      expBox.innerHTML = '<span>' + DEFAULT_MSG + '</span>';
    }}
  }}

  root.querySelectorAll('[data-k]').forEach(el => {{
    const k = el.getAttribute('data-k');
    el.addEventListener('mouseenter', () => applyHighlight(k));
    el.addEventListener('mouseleave', () => applyHighlight(pinnedKey));
    el.addEventListener('click', (e) => {{
      e.stopPropagation();
      pinnedKey = (pinnedKey === k) ? null : k;
      applyHighlight(pinnedKey || k);
    }});
  }});
}})();
</script>
</body>
</html>"""

    components.html(page_html, height=height, scrolling=True)
