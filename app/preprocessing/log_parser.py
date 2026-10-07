"""
Log Parser
==========
Parses raw Apache Combined Log Format lines into HTTPLogEntry objects.

Apache Combined Log Format:
  IP - - [DD/Mon/YYYY:HH:MM:SS +ZZZZ] "METHOD /path HTTP/1.x" STATUS SIZE "referrer" "user-agent" response_time

The last field (response_time ms) is non-standard but present in logfiles.log.
"""

from __future__ import annotations

import re
from typing import List, Optional

from app.models.schemas import HTTPLogEntry

# ── Regex for Apache Combined Log + optional trailing response_time ───────────
_LOG_RE = re.compile(
    r'(?P<ip>\S+)'                          # client IP
    r'\s+\S+\s+\S+\s+'                      # ident, auth (usually - -)
    r'\[(?P<timestamp>[^\]]+)\]'            # [timestamp]
    r'\s+"(?P<method>\S+)'                  # "METHOD
    r'\s+(?P<path>\S+)'                     # /path
    r'\s+\S+"'                              # HTTP/x.x"
    r'\s+(?P<status>\d{3})'                 # status code
    r'\s+(?P<size>\d+)'                     # response size
    r'(?:\s+"(?P<referrer>[^"]*)")?'        # optional "referrer"
    r'(?:\s+"(?P<user_agent>[^"]*)")?'      # optional "user-agent"
    r'(?:\s+(?P<response_time>\d+))?'       # optional response_time (ms)
)


def parse_log_line(line: str) -> Optional[HTTPLogEntry]:
    """
    Parse a single Apache Combined Log line.
    Returns None if the line does not match the expected format.
    """
    m = _LOG_RE.match(line.strip())
    if not m:
        return None

    referrer = m.group("referrer")
    if referrer in ("-", ""):
        referrer = None

    rt = m.group("response_time")

    return HTTPLogEntry(
        ip=m.group("ip"),
        timestamp=m.group("timestamp"),
        method=m.group("method").upper(),
        path=m.group("path"),
        status_code=int(m.group("status")),
        response_size=int(m.group("size")),
        referrer=referrer,
        user_agent=m.group("user_agent"),
        response_time=float(rt) if rt else None,
    )


def parse_log_lines(lines: List[str]) -> List[HTTPLogEntry]:
    """
    Parse a list of raw log strings.
    Silently skips lines that cannot be parsed.
    """
    entries: List[HTTPLogEntry] = []
    for line in lines:
        entry = parse_log_line(line)
        if entry:
            entries.append(entry)
    return entries
