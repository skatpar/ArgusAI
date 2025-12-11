"""
Common styling utilities for ArgusAI dashboard
Professional theme with #744ada, black, grey color palette
"""

def get_status_indicator(status_type):
    """Get professional status indicators without emojis"""
    indicators = {
        'success': '<span style="color: #744ada; font-weight: 600;">●</span>',
        'warning': '<span style="color: #666666; font-weight: 600;">●</span>',
        'error': '<span style="color: #000000; font-weight: 600;">●</span>',
        'info': '<span style="color: #cccccc; font-weight: 600;">●</span>',
        'good': '<span style="background-color: #744ada; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85rem; font-weight: 500;">GOOD</span>',
        'monitor': '<span style="background-color: #666666; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85rem; font-weight: 500;">MONITOR</span>',
        'alert': '<span style="background-color: #000000; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.85rem; font-weight: 500;">ALERT</span>',
    }
    return indicators.get(status_type, '')


def format_header(text, level=1):
    """Format headers consistently"""
    if level == 1:
        return f'<p class="main-header">{text}</p>'
    elif level == 2:
        return f'<p class="sub-header">{text}</p>'
    else:
        return f'<h{level} style="color: #2b2b2b; font-weight: 600;">{text}</h{level}>'


def get_severity_badge(severity):
    """Get severity badge without emojis"""
    badges = {
        'Critical': '<span style="background-color: #000000; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">CRITICAL</span>',
        'High': '<span style="background-color: #2b2b2b; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">HIGH</span>',
        'Warning': '<span style="background-color: #666666; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">WARNING</span>',
        'Medium': '<span style="background-color: #666666; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">MEDIUM</span>',
        'Low': '<span style="background-color: #cccccc; color: #2b2b2b; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600;">LOW</span>',
        'Info': '<span style="background-color: #f5f5f5; color: #2b2b2b; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; border: 1px solid #cccccc;">INFO</span>',
    }
    return badges.get(severity, severity)


def get_color_palette():
    """Return the standard color palette"""
    return {
        'primary': '#744ada',
        'dark_primary': '#5a38ad',
        'light_primary': '#9b7fe8',
        'black': '#000000',
        'dark_grey': '#2b2b2b',
        'medium_grey': '#666666',
        'light_grey': '#cccccc',
        'bg_grey': '#f5f5f5',
        'white': '#ffffff'
    }
