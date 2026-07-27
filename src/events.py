from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Event:
    """Class for keeping track of an event."""
    type: str
    timestamp: datetime
    metadata: dict = field(default_factory = dict)