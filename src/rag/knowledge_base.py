import chromadb
from sentence_transformers import SentenceTransformer

class BirthDefectRAG:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("ncrna_birth_defect")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量编码器
    
    def add_records(self, records: list[dict]):
        """
        records: [{"id": "clinvar_123", "text": "lncRNA H19 variant associated with ...", "metadata": {...}}]
        """
        texts = [r["text"] for r in records]
        embeddings = self.encoder.encode(texts).tolist()
        
        self.collection.add(
            ids=[r["id"] for r in records],
            embeddings=embeddings,
            documents=texts,
            metadatas=[r["metadata"] for r in records]
        )
    
    def retrieve(self, query_text: str, n_results=3):
        """检索与查询序列/变异相关的已知出生缺陷记录"""
        query_emb = self.encoder.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n_results
        )
        return results