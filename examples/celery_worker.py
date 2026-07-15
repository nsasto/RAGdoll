"""Minimal Celery worker entry point for deployment_scaled.yaml."""

import asyncio
import os

from celery import Celery

from ragdoll import Ragdoll
from ragdoll.errors import JobLeaseUnavailableError

CONFIG_PATH = os.environ.get("RAGDOLL_CONFIG", "examples/deployment_scaled.yaml")
celery_app = Celery("ragdoll", broker=os.environ["CELERY_BROKER_URL"])
rag = Ragdoll.from_config(CONFIG_PATH)


@celery_app.task(
    name="ragdoll.execute_ingestion",
    acks_late=True,
    autoretry_for=(JobLeaseUnavailableError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 12},
)
def execute_ingestion(job_id: str) -> None:
    """Workers receive durable identifiers; state remains in PostgreSQL."""
    asyncio.run(rag.durable_ingestion.execute(job_id))
