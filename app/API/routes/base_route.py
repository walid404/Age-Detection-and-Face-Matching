from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Health check endpoint.

    This endpoint is used to verify that the API service is running
    and responsive. It can be consumed by load balancers, monitoring
    systems, or orchestration tools (e.g., Docker, Kubernetes).

    Returns
    -------
    dict
        A dictionary containing the service health status and name.

        Example
        -------
        {
            "status": "ok",
            "service": "face matcher and age prediction API"
        }
    """
    return {
        "status": "ok",
        "service": "face matcher and age prediction API"
    }