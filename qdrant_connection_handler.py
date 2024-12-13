from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import pandas as pd
from typing import Callable

class QDrantHandler:
    def __init__(self, url: str = None, in_memory: bool = False):
        if in_memory:
            # Initialize Qdrant in-memory
            self.qdrant_client = QdrantClient(path=":memory:")
        elif url:
            # Initialize Qdrant with a URL
            self.qdrant_client = QdrantClient(url=url)
        else:
            raise ValueError("Either 'url' must be provided or 'in_memory' must be True")

        # Create/recreate the collection
        self.qdrant_client.recreate_collection(
            collection_name="translations_collection",
            vectors_config={"size": 384, "distance": "Cosine"}
        )
    
    def store_embeddings_in_qdrant(self, df: pd.DataFrame, embedding_model):
        if embedding_model is None:
            raise Exception("No Embedding model passed")
        
        if df.empty:
            raise ValueError("DataFrame empty")
        
        point_id = 0
        all_points = []

        for idx, row in df.iterrows():
            english_embeddings = embedding_model.get_embeddings(row['English_chunks'])
            french_embeddings = embedding_model.get_embeddings(row['French_chunks'])

            for eng_emb, fr_emb in zip(english_embeddings, french_embeddings):
                point_id += 1
                point = PointStruct(
                    id=point_id,
                    vector=eng_emb.tolist(),
                    payload={"language": "english", "text": row['en'], "translation": row['fr']}
                )
                all_points.append(point)

                point_id += 1
                point = PointStruct(
                    id=point_id,
                    vector=fr_emb.tolist(),
                    payload={"language": "french", "text": row['fr'], "translation": row['en']}
                )
                all_points.append(point)

        self.qdrant_client.upsert(
            collection_name="translations_collection",
            points=all_points
        )
        
    def query_qdrant(self, query_text, embedding_model):
        query_embedding = embedding_model.get_embeddings([query_text])[0]

        search_results = self.qdrant_client.search(
            collection_name="translations_collection",
            query_vector=query_embedding.tolist(),
            limit=5
        )

        result_translations = []
        for result in search_results:
            payload = result.payload

            if payload['language'] == 'english' and query_text.lower() != payload['text'].lower():
                result_translations.append(payload['translation'])
            elif payload['language'] == 'french' and query_text.lower() != payload['text'].lower():
                result_translations.append(payload['translation'])

        return result_translations
