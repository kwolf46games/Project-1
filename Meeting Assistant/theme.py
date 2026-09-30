"""Shared colors and window setup for the overlay and the bank manager."""

BG, PANEL, FG, MUTED = "#15171c", "#1f232b", "#e8eaf0", "#8a91a0"
ACCENT, GOOD, WARN, BAD = "#6ea8fe", "#4cc38a", "#e5b454", "#e5675f"
PLACEHOLDER_BG = "#5a4712"
SELECT_BG, HOVER_BG = "#2f4a7a", "#2b303a"


def enable_dpi_awareness() -> None:
    """Crisp text on scaled (high-DPI) displays. Must run before the first Tk window."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
