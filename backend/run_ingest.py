# backend/run_ingest.py
import sys
sys.path.append("app")   # taaki internal imports (config, retrievers waghera) resolve ho jaayein

from app.retrievers.ingest import ingest_files

files = [
    "data/raw/Hands On Machine Learning with Scikit Learn and TensorFlow.pdf",
    # chahe to baaki 2 PDFs bhi add kar sakta hai:
    # "data/raw/final-review.pdf",
    # "data/raw/research-paper.pdf",
]

result = ingest_files(files)
print(f"\n✅ Ingestion complete: {result}")