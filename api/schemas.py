from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None

class LLMReply(BaseModel):
    text: str
    timestamp: datetime

class ASRLLMReply(LLMReply):
    transcript: Optional[str] = None
