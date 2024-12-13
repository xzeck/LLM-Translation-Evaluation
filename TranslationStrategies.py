from ITranslationStrategy import TranslationStrategy
from LLM import LLM
from RAG import RAG
from qdrant_connection_handler import QDrantHandler

class LLMTranslationStrategy(TranslationStrategy):
    def __init__(self, arch: LLM):
        self.arch = arch
    
    def translate(self, query_text, embedding_model):
        return self.arch.translate_with_llm(query_text)
    
    
class RAGTranslationStrategy(TranslationStrategy):
    def __init__(self, arch: RAG, qdrant_handler: QDrantHandler):
        self.arch = arch
        self.qdrant_handler = qdrant_handler
    
    def translate(self, query_text, embedding_model):
        related_translations = self.qdrant_handler.query_qdrant(query_text, embedding_model)
        return self.arch.translate_with_llm(query_text, related_translations)
    
    
    