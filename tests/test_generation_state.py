import pytest

from ragdoll.contracts import CorpusId, GenerationId, TenantId
from ragdoll.errors import (
    GenerationActiveError,
    GenerationNotFoundError,
    TenantIsolationError,
)
from ragdoll.generation_state import (
    FileGenerationStateStore,
    GenerationRecord,
    MemoryGenerationStateStore,
)


def record(item_id, tenant="acme", corpus="docs"):
    return GenerationRecord(
        id=GenerationId(item_id),
        tenant=TenantId(tenant),
        corpus=CorpusId(corpus),
        vector_ids=(f"vector-{item_id}",),
        document_count=1,
        checksum=f"checksum-{item_id}",
    )


@pytest.fixture(params=["memory", "file"])
def state_store(request, tmp_path):
    if request.param == "memory":
        return MemoryGenerationStateStore()
    return FileGenerationStateStore(tmp_path / "state.json")


@pytest.mark.asyncio
async def test_generation_state_contract_promote_and_rollback(state_store):
    await state_store.put(record("one"))
    await state_store.put(record("two"))

    await state_store.promote(GenerationId("one"), tenant="acme", corpus="docs")
    await state_store.promote(GenerationId("two"), tenant="acme", corpus="docs")
    restored = await state_store.rollback("acme", "docs")

    assert restored == GenerationId("one")
    assert await state_store.active("acme", "docs") == GenerationId("one")


@pytest.mark.asyncio
async def test_generation_state_contract_enforces_tenant(state_store):
    await state_store.put(record("one"))

    with pytest.raises(TenantIsolationError):
        await state_store.promote(GenerationId("one"), tenant="other")


@pytest.mark.asyncio
async def test_generation_state_contract_unknown_generation(state_store):
    with pytest.raises(GenerationNotFoundError):
        await state_store.get(GenerationId("missing"))


@pytest.mark.asyncio
async def test_generation_state_discards_only_unpublished_generation(state_store):
    await state_store.put(record("active"))
    await state_store.put(record("staged"))
    await state_store.promote(GenerationId("active"))

    await state_store.delete_generation(GenerationId("staged"))

    with pytest.raises(GenerationNotFoundError):
        await state_store.get(GenerationId("staged"))
    with pytest.raises(GenerationActiveError):
        await state_store.delete_generation(GenerationId("active"))


@pytest.mark.asyncio
async def test_file_generation_state_is_shared_across_instances(tmp_path):
    path = tmp_path / "state.json"
    writer = FileGenerationStateStore(path)
    await writer.put(record("one"))
    await writer.promote(GenerationId("one"))

    reader = FileGenerationStateStore(path)

    assert await reader.active("acme", "docs") == GenerationId("one")
