from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.assistant import Assistant
from rag_assistant.evaluation import evaluate_assistant, evaluate_retrieval, load_questions
from rag_assistant.retrieval import build_retriever


def test_api_respects_group_header():
    api.get_assistant.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    q = {"question": "When are salaries paid each month?"}
    public = client.post("/ask", json=q).json()
    assert all(not c["chunk_id"].startswith("payroll") for c in public["citations"])
    finance = client.post("/ask", json=q, headers={"X-User-Groups": "finance"}).json()
    assert finance["citations"][0]["chunk_id"].startswith("payroll-schedule")
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_evaluation_runs_and_has_no_leaks(chunks):
    qs = load_questions("eval/questions.jsonl")
    m = evaluate_retrieval(build_retriever("bm25"), chunks, qs)
    assert 0.5 < m.hit_at_3 <= 1.0
    r = evaluate_assistant(Assistant.from_settings(), qs)
    assert r["access_control_leaks"] == 0
    assert r["injection_block_rate"] == 1.0
