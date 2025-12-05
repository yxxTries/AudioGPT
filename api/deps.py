from __future__ import annotations
from typing import Optional
from asr.service import ASRService
from llm.service import LLMService

_asr_service: Optional[ASRService] = None
_llm_service: Optional[LLMService] = None

def get_asr_service() -> ASRService:
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service

def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
