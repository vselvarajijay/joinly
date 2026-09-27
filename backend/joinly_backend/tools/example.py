from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from joinly_backend.registry import ToolContext, tool


@tool
async def get_current_time(timezone: str, context: ToolContext) -> str:  # noqa: ARG001
    """Get the current date and time in a timezone.

    Args:
        timezone: IANA timezone name, e.g. "Europe/Berlin" or "America/New_York".
        context: Meeting context (not exposed to the model).
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return f"Unknown timezone: {timezone}"
    return now.strftime("%A, %d %B %Y, %H:%M %Z")
