"""Inline SVG section-heading icons for the Gradio interface."""

from html import escape


_ICON_PATHS = {
    "automation": "M12 3v3m0 12v3M4.2 4.2l2.1 2.1m11.4 11.4 2.1 2.1M3 12h3m12 0h3M4.2 19.8l2.1-2.1m11.4-11.4 2.1-2.1M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z",
    "dataset": "M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v13a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 18.5v-13ZM8 8h8M8 12h8M8 16h5",
    "export": "M12 3v12m0 0 4-4m-4 4-4-4M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4",
    "preview": "M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Zm9.5 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z",
    "settings": "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0-5 1 2.5 2.7.5-.9 2.6 1.9 1.9-1.9 1.9.9 2.6-2.7.5L12 21l-1-2.5-2.7-.5.9-2.6-1.9-1.9 1.9-1.9-.9-2.6 2.7-.5L12 3Z",
    "training": "M4 19V5m0 14h16M8 15v-4m4 4V7m4 8v-6",
    "upload": "M12 16V4m0 0 4 4m-4-4-4 4M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4",
    "workflow": "M6 3h12v5H6zM4 16h6v5H4zm10 0h6v5h-6zm-5-8v5m0 0H7v3m5-3h5v3",
}


def section_heading(title: str, icon: str, level: int = 3, divider: bool = False) -> str:
    """Build an accessible heading with a consistent SVG icon.

    Args:
        title: Localized heading text.
        icon: Name of the icon to render.
        level: HTML heading level from 1 through 6.
        divider: Whether to add a horizontal divider before the heading.

    Returns:
        Escaped HTML for a styled section heading.
    """
    if not 1 <= level <= 6:
        raise ValueError("Heading level must be between 1 and 6.")
    path = _ICON_PATHS.get(icon, _ICON_PATHS["settings"])
    divider_html = "<hr>" if divider else ""
    return (
        f'{divider_html}<h{level} class="ui-section-heading">'
        '<svg class="ui-section-icon" aria-hidden="true" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.8"><path d="{path}"/></svg>'
        f"{escape(title)}</h{level}>"
    )
