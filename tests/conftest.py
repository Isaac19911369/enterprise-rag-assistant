import pytest

from rag_assistant.ingest import load_corpus


@pytest.fixture(scope="session")
def chunks():
    return load_corpus("corpus/northwind")
