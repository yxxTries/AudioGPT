
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ASRSegment:
    start: float
    end: float
    text: str

@dataclass
class ASRResult:
    text: str
    language: Optional[str]
    segments: List[ASRSegment]
    duration: Optional[float]
