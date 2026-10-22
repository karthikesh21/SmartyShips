"""
Unit tests for the SmartyShips Chart Theming module.
Verifies dynamic palette selection and Plotly layout customization in light and dark modes.
"""

import pytest
import plotly.graph_objects as go
from src.chart_theme import get_theme_palette, apply_smarty_theme, THEME_COLORS

def test_theme_palette_keys():
    """Verify both light and dark palettes contain required color tokens."""
    for mode in [True, False]:
        palette = get_theme_palette(mode)
        assert "bg_canvas" in palette
        assert "bg_card" in palette
        assert "text_primary" in palette
        assert "text_secondary" in palette
        assert "plotly_template" in palette
        assert "primary_blue" in palette
        assert "emerald_green" in palette
        assert "amber_orange" in palette
        assert "crimson_red" in palette

def test_theme_palette_template_selection():
    """Verify correct template assignment for dark and light modes."""
    dark_pal = get_theme_palette(True)
    light_pal = get_theme_palette(False)
    assert dark_pal["plotly_template"] == "plotly_dark"
    assert light_pal["plotly_template"] == "plotly_white"

def test_apply_smarty_theme_dark_mode():
    """Verify apply_smarty_theme configures font colors, title, and transparent backgrounds for dark mode."""
    fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[10, 20, 30]))
    styled_fig = apply_smarty_theme(fig, dark_mode=True, title="Transit Performance")

    assert styled_fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert styled_fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert styled_fig.layout.title.text == "Transit Performance"
    assert styled_fig.layout.title.font.color == THEME_COLORS["dark"]["text_primary"]
    assert styled_fig.layout.template is not None

def test_apply_smarty_theme_light_mode():
    """Verify apply_smarty_theme configures font colors and transparent backgrounds for light mode."""
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    styled_fig = apply_smarty_theme(fig, dark_mode=False, title="Corridor Distribution")

    assert styled_fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert styled_fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert styled_fig.layout.title.text == "Corridor Distribution"
    assert styled_fig.layout.title.font.color == THEME_COLORS["light"]["text_primary"]
    assert styled_fig.layout.template is not None
