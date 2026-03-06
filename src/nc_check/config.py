from __future__ import annotations

from pydantic import BaseModel


class NcCheckConfig(BaseModel):
    GCB_YEAR: int = 2025


settings = NcCheckConfig()
