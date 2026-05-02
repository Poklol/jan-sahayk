import sys
import os
sys.path.append('.')

from orchestrator import JanSahayakOrchestrator
from rag import RAGAgent

class MockWebAgent:
    def search_official(self, q, max_results=1):
        return []

rag_agent = RAGAgent(docs_dir=os.path.join(os.getcwd(), 'docs'))
rag_agent.ensure_index()

o = JanSahayakOrchestrator(rag_agent=rag_agent, web_agent=MockWebAgent())
# Inject an API key to allow Gemini to work!
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

profile = {
    "name": "Charukesh",
    "occupation": "student",
    "income": "3 lakh",
    "loan_amount": "1 lakh",
    "state": "Tamil Nadu"
}

print("Running query...")
res = o.handle_query("I need a scholarship", profile)

print("Retrieval Evidence (Count):", len(res.get("rag_evidence", [])))
for i, chunk in enumerate(res.get("rag_evidence", [])):
    print(f" Chunk {i+1} [Score: {chunk.get('score', 0):.2f}] - {chunk.get('source', '')}")

print("\n--- FINAL ANSWER ---")
# Because it is a generator stream, we must eagerly consume it for the test
stream = res.get("response_markdown")
import sys
sys.stdout.reconfigure(encoding='utf-8')
if hasattr(stream, '__iter__') and not isinstance(stream, str):
    print("".join(list(stream)))
else:
    print(stream)
