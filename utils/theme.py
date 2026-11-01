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

# EDA chart palette (validated for colour-vision deficiency, light surface).
# Categorical slots are assigned in fixed order and capped at 3 series per
# chart (campaigns and engagement levels both have exactly 3).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"negative": "#e34948", "neutral": "#f0efec", "positive": "#2a78d6"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984",
       "grid": "#ecebe7", "surface": "#fcfcfb"}

# Plotly template used by dashboard charts
CHART_TEMPLATE = "plotly_dark"
