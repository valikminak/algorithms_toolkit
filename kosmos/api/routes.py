import time
from fastapi import APIRouter, Path, Body, HTTPException

from kosmos.core.registry import AlgorithmRegistry
from kosmos.api.schemas import (
    AlgorithmInfo,
    AlgorithmResponse,
    AlgorithmRequest,
    AlgorithmListResponse,
    DomainListResponse
)

router = APIRouter()


@router.get("/domains", response_model=DomainListResponse)
async def get_domains():
    """Get all available algorithm domains."""
    registry = AlgorithmRegistry()
    domains = registry.get_domains()
    return {"domains": domains}


@router.get("/domains/{domain_id}/algorithms", response_model=AlgorithmListResponse)
async def get_algorithms(domain_id: str = Path(..., description="Domain ID")):
    """Get all algorithms for a specific domain."""
    registry = AlgorithmRegistry()

    if domain_id not in registry.get_domains():
        raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")

    algorithms = []
    for algo_id in registry.get_algorithm_ids(domain_id):
        try:
            algorithm = registry.get_algorithm(domain_id, algo_id)
            meta = algorithm.metadata
            algorithms.append(AlgorithmInfo(
                id=meta["id"],
                name=meta["name"],
                category=meta["category"],
                description=meta["description"]
            ))
        except Exception as e:
            print(f"Error loading algorithm {algo_id}: {e}")

    return {"algorithms": algorithms}


@router.post("/domains/{domain_id}/algorithms/{algorithm_id}/run", response_model=AlgorithmResponse)
async def run_algorithm(
        domain_id: str = Path(..., description="Domain ID"),
        algorithm_id: str = Path(..., description="Algorithm ID"),
        request: AlgorithmRequest = Body(...)
):
    """Run a specific algorithm and return results with visualization data."""
    registry = AlgorithmRegistry()

    try:
        # Get algorithm instance
        algorithm = registry.get_algorithm(domain_id, algorithm_id)

        # Run algorithm
        input_data = request.input
        start_time = time.time()
        result = algorithm.execute(input_data)
        execution_time = time.time() - start_time

        # Generate visualization frames
        frames = algorithm.get_visualization_frames(input_data)

        return AlgorithmResponse(
            algorithm=algorithm_id,
            domain=domain_id,
            input=input_data,
            output=result,
            execution_time=execution_time,
            visualization=frames,
            metadata=algorithm.metadata
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Algorithm {algorithm_id} in domain {domain_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing algorithm: {str(e)}")