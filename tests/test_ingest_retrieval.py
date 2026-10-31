import pytest

from rag_assistant.ingest import parse_document
from rag_assistant.retrieval import RETRIEVERS, build_retriever


def test_chunks_carry_metadata(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    payroll = [c for c in chunks if c.doc == "payroll-schedule"]
    assert payroll and all(c.groups == ("finance",) for c in payroll)
    assert all(c.text.startswith(c.title) for c in chunks)


def test_long_sections_are_split_with_overlap(tmp_path):
    body = " ".join(f"w{i}" for i in range(500))
    f = tmp_path / "doc.md"
    f.write_text(f"---\ntitle: T\n---\n# T\n\n## Big\n{body}\n")
    parts = parse_document(f, max_words=200, overlap=50)
    assert len(parts) == 3
    assert parts[1].id == "doc#big-1"
    assert "w150" in parts[0].text and "w150" in parts[1].text  # overlap


@pytest.mark.parametrize("name", [r for r in RETRIEVERS if "dense" not in r])
def test_retrievers_find_obvious_section(name, chunks):
    r = build_retriever(name).fit(chunks)
    top = r.search("hotel nightly limit in Munich", k=3)
    assert top[0].chunk.id == "travel-and-expenses#hotels"


def test_access_control_filters_restricted_documents(chunks):
    r = build_retriever("hybrid").fit(chunks)
    q = "When are salaries paid each month?"
    public = r.search(q, k=10, groups={"all"})
    assert all(h.chunk.doc != "payroll-schedule" for h in public)
    finance = r.search(q, k=1, groups={"all", "finance"})
    assert finance[0].chunk.doc == "payroll-schedule"
