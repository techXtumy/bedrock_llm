from pydantic import BaseModel, Field
from typing import Optional, List


class ModelConfig(BaseModel):
    """
    Note:
    Attributes:
        repetition_penalty: Only for Jamba Model.
        presence_penalty: Only for Jamba Model.
        number_of_response: Only for Jamba Model.
    """
    temperature: float = Field(default=0, ge=0, le=1)
    max_tokens: int = Field(default=2024, gt=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    top_k: Optional[int] = Field(default=60)
    stop_sequences: List[str] = Field(default_factory=list)
    number_of_responses: int = Field(default=1, ge=1, le=16)