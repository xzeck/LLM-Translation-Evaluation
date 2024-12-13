import pandas as pd
import re
from typing import Callable

class PreProcessor:
    
    def __init__(self):
        self.df = pd.DataFrame()
        pass
    
    def read(self, file="split.csv", nrows=900):
        # Load and preprocess data
        try:
            self.df = pd.read_csv(file, nrows=nrows)
            self.df['en'] = self.df['en'].str.strip().str.lower()
            self.df['fr'] = self.df['fr'].str.strip().str.lower()
        except Exception as e:
            raise Exception("Error while reading file")
        
    def default_chunker(self, text) -> list:
        text = str(text)
        sentences = re.split(r'(?<=\!|\?|\.|\|)(\s*)', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def chunk_sentence(self, chunker: Callable[str, list]) -> pd.DataFrame :
        
        if self.df.empty:
            raise ValueError("Error, no data in csv")
        
        chunker = chunker or self.default_chunker
        
        self.df['English_chunks'] = self.df['en'].apply(chunker)
        self.df['French_chunks'] = self.df['fr'].apply(chunker)
        
        return self.df
