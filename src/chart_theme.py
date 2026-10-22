"""
Chart theming and styling utility for SmartyShips.
Provides centralized Plotly styling for consistent, high-contrast rendering
across both Light and Dark modes.
"""

from typing import Dict, Any
import plotly.graph_objects as go

THEME_COLORS = {
    "dark": {
        "bg_canvas": "#0b0f19",
        "bg_card": "#111827",
        "bg_subtle": "#1f2937",
        "text_primary": "#f8fafc",
        "text_secondary": "#cbd5e1",
        "text_muted": "#94a3b8",
        "border": "#374151",
        "grid": "rgba(255, 255, 255, 0.08)",
        "zero_line": "rgba(255, 255, 255, 0.2)",
        "primary_blue": "#38bdf8",
        "accent_cyan": "#06b6d4",
        "emerald_green": "#34d399",
        "amber_orange": "#fbbf24",
        "crimson_red": "#f87171",
        "purple_violet": "#a78bfa",
        "plotly_template": "plotly_dark",
    },
    "light": {
        "bg_canvas": "#f8fafc",
        "bg_card": "#ffffff",
        "bg_subtle": "#f1f5f9",
        "text_primary": "#0f2744",
        "text_secondary": "#334155",
        "text_muted": "#64748b",
        "border": "#e2e8f0",
        "grid": "rgba(0, 0, 0, 0.06)",
        "zero_line": "rgba(0, 0, 0, 0.25)",
        "primary_blue": "#0284c7",
        "accent_cyan": "#0891b2",
        "emerald_green": "#059669",
        "amber_orange": "#d97706",
        "crimson_red": "#dc2626",
        "purple_violet": "#7c3aed",
        "plotly_template": "plotly_white",
    }
}

def get_theme_palette(dark_mode: bool) -> Dict[str, Any]:
    """Returns the color token dictionary for the active theme."""
    return THEME_COLORS["dark"] if dark_mode else THEME_COLORS["light"]

def apply_smarty_theme(fig: go.Figure, dark_mode: bool, title: str = "") -> go.Figure:
    """
    Applies comprehensive styling to any Plotly figure so that titles, axes,
    legends, gridlines, and hover labels are crisp and readable in both modes.
    """
    p = get_theme_palette(dark_mode)

    layout_updates: Dict[str, Any] = {
        "template": p["plotly_template"],
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            "color": p["text_secondary"],
            "size": 12,
        },
        "hoverlabel": {
            "bgcolor": p["bg_card"],
            "font": {
                "color": p["text_primary"],
                "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "size": 12,
            },
            "bordercolor": p["border"],
        },
        "legend": {
            "font": {"color": p["text_primary"], "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
        },
    }

    if title:
        layout_updates["title"] = {
            "text": title,
            "font": {
                "color": p["text_primary"],
                "size": 14,
                "weight": 700,
            },
            "x": 0.02,
            "xanchor": "left",
        }
    elif fig.layout.title and fig.layout.title.text:
        layout_updates["title"] = {
            "text": fig.layout.title.text,
            "font": {
                "color": p["text_primary"],
                "size": 14,
                "weight": 700,
            },
            "x": 0.02,
            "xanchor": "left",
        }

    fig.update_layout(**layout_updates)

    # Style axes
    fig.update_xaxes(
        color=p["text_secondary"],
        title_font={"color": p["text_primary"], "size": 12},
        tickfont={"color": p["text_secondary"], "size": 11},
        gridcolor=p["grid"],
        zerolinecolor=p["zero_line"],
        showline=True,
        linecolor=p["border"],
    )
    fig.update_yaxes(
        color=p["text_secondary"],
        title_font={"color": p["text_primary"], "size": 12},
        tickfont={"color": p["text_secondary"], "size": 11},
        gridcolor=p["grid"],
        zerolinecolor=p["zero_line"],
        showline=True,
        linecolor=p["border"],
    )

    # Style coloraxes if present (e.g. scatter plots, heatmaps)
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=p["text_secondary"], size=10),
            title_font=dict(color=p["text_primary"], size=11),
        )
    )

    return fig
