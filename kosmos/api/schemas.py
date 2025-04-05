from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AlgorithmMetadata(BaseModel):
    id: str
    name: str
    category: str
    time_complexity: Dict[str, str]
    space_complexity: str
    stable: bool
    description: str

class AlgorithmInfo(BaseModel):
    id: str
    name: str
    category: str
    description: str

class DomainListResponse(BaseModel):
    domains: List[str]

class AlgorithmListResponse(BaseModel):
    algorithms: List[AlgorithmInfo]

class AlgorithmRequest(BaseModel):
    input: List[int]
    options: Optional[Dict[str, Any]] = None

class AlgorithmResponse(BaseModel):
    algorithm: str
    domain: str
    input: List[int]
    output: List[int]
    execution_time: float
    visualization: List[Dict[str, Any]]
    metadata: AlgorithmMetadata