import pytest

from ragdoll.ingestion import clear_loader_registry, register_loader, register_loader_class, get_loader


class Dummy:
    pass


def test_registry_normalization_and_lookup():
    clear_loader_registry()

    # Register via decorator with short name
    @register_loader('.PDF')
    class PdfLoader:
        pass

    # Programmatic registration with dot
    register_loader_class('.md', Dummy)

    # Lookups should be normalized
    assert get_loader('pdf') is PdfLoader
    assert get_loader('.pdf') is PdfLoader
    assert get_loader('PDF') is PdfLoader

    assert get_loader('md') is Dummy
    assert get_loader('.md') is Dummy

    clear_loader_registry()
