"""
Input validation utilities for the API.

Centralizes all request validation logic extracted from app.py.
"""

VALID_CAMPAIGNS = [
    "Brand Awareness", "Product Launch", "Holiday Sale",
    "Retargeting", "Lead Generation", "App Install",
]
VALID_CHANNELS = [
    "Google Ads", "Facebook", "Instagram", "YouTube", "Email", "LinkedIn",
]
VALID_DEVICES = ["Mobile", "Desktop", "Tablet"]


def validate_predict_input(data: dict) -> list:
    """
    Validate prediction input and return list of errors.

    Parameters
    ----------
    data : dict
        Raw JSON input from the request body.

    Returns
    -------
    list
        List of error message strings. Empty if valid.
    """
    errors = []

    required = [
        "campaign", "channel", "device", "impressions", "clicks",
        "spend", "conversions", "bounce_rate", "avg_session_duration",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        errors.append(f"Missing required fields: {missing}")
        return errors

    if data.get("campaign") not in VALID_CAMPAIGNS:
        errors.append(f"Invalid campaign. Valid: {VALID_CAMPAIGNS}")
    if data.get("channel") not in VALID_CHANNELS:
        errors.append(f"Invalid channel. Valid: {VALID_CHANNELS}")
    if data.get("device") not in VALID_DEVICES:
        errors.append(f"Invalid device. Valid: {VALID_DEVICES}")

    for field in ["impressions", "clicks", "spend", "conversions", "avg_session_duration"]:
        val = data.get(field)
        if val is not None and (not isinstance(val, (int, float)) or val < 0):
            errors.append(f"{field} must be a non-negative number")

    bounce = data.get("bounce_rate")
    if bounce is not None and (not isinstance(bounce, (int, float)) or bounce < 0 or bounce > 1):
        errors.append("bounce_rate must be between 0 and 1")

    clicks = data.get("clicks", 0)
    impressions = data.get("impressions", 0)
    if isinstance(clicks, (int, float)) and isinstance(impressions, (int, float)):
        if clicks > impressions:
            errors.append("clicks cannot exceed impressions")

    return errors
