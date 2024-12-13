from abc import ABC, abstractmethod


class TranslationStrategy(ABC):
    @abstractmethod
    def translate(self, query_text, embedding_model, qdrant_handler):
        pass
    