from pydantic import BaseModel, Field
from typing import Optional, List


class ModelConfig(BaseModel):
    temperature: float = Field(default=0, ge=0, le=1)
    max_tokens: int = Field(default=2024, gt=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    top_k: Optional[int] = Field(default=60)
    stop_sequences: List[str] = Field(default_factory=list)