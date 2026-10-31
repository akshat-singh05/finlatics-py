"""
Shared design system constants for Finlatics-Py.

Centralizes PALETTE, COLORS, and chart configuration used across
all visualization modules. Import from here instead of defining locally.

Usage:
    from utils.theme import PALETTE, COLORS
"""

# Ordered color palette for charts (up to 6 categories)
PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6", "#8B5CF6"]

# Semantic color mapping
COLORS = {
    "primary": "#6366F1",
    "success": "#10B981",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "gold": "#D97706",
}

# Plotly template used by dashboard charts
CHART_TEMPLATE = "plotly_dark"
