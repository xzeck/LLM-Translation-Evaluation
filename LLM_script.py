import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import re
from transformers import BitsAndBytesConfig
from langdetect import detect, DetectorFactory

print(torch.cuda.is_available())

df = pd.read_csv("split.csv")
df['en'] = df['en'].str.strip().str.lower()
df['fr'] = df['fr'].str.strip().str.lower()
def chunk_sentences(text):
    text = str(text)
    sentences = re.split(r'(?<=\!|\?|\.|\|)(\s*)', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

df['English_chunks'] = df['en'].apply(chunk_sentences)
df['French_chunks'] = df['fr'].apply(chunk_sentences)

df.head()


class EmbeddingModel:
    def __init__(self, model_name="thenlper/gte-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        return embeddings.numpy()

embedding_model = EmbeddingModel()

df.head()

qdrant_client = QdrantClient(path=":memory:")

qdrant_client.recreate_collection(
    collection_name="translations_collection",
    vectors_config={"size": 384, "distance": "Cosine"} 
)

def store_embeddings_in_qdrant(df, embedding_model):
    point_id = 0
    all_points = []

    # Loop over the English and French chunks to generate embeddings and store them
    for idx, row in df.iterrows():
        # Get embeddings for English and French chunks
        english_embeddings = embedding_model.get_embeddings(row['English_chunks'])
        french_embeddings = embedding_model.get_embeddings(row['French_chunks'])

        # Store both English and French embeddings in Qdrant
        for eng_emb, fr_emb in zip(english_embeddings, french_embeddings):
            # Create a unique point ID for the English text
            point_id += 1
            point = PointStruct(
                id=point_id,
                vector=eng_emb.tolist(),  
                payload={"language": "english", "text": row['en'], "translation": row['fr']}
            )
            all_points.append(point)

            # Create a unique point ID for the French text
            point_id += 1
            point = PointStruct(
                id=point_id,
                vector=fr_emb.tolist(),  
                payload={"language": "french", "text": row['fr'], "translation": row['en']}
            )
            all_points.append(point)

    # Insert all points into Qdrant collection
    qdrant_client.upsert(
        collection_name="translations_collection",
        points=all_points
    )


store_embeddings_in_qdrant(df, embedding_model)

def query_qdrant(query_text, embedding_model):
    # Get query embedding
    query_embedding = embedding_model.get_embeddings([query_text])[0]

    # Search Qdrant for the most similar points
    search_results = qdrant_client.search(
        collection_name="translations_collection",
        query_vector=query_embedding.tolist(),
        limit=5 
    )

    # Prepare the results to return
    result_translations = []
    for result in search_results:
        # Extract the text and its translation from the payload
        payload = result.payload

        if payload['language'] == 'english' and query_text.lower() != payload['text'].lower():
            result_translations.append(payload['translation'])  # French translation
        elif payload['language'] == 'french' and query_text.lower() != payload['text'].lower():
            result_translations.append(payload['translation'])  # English translation

    # # Only return French translations when the query is in English
    #     if payload['language'] == 'english' and query_text.lower() != payload['text'].lower():
    #         result_translations.append(payload['translation'])  # French translation

    return result_translations


result_translations = query_qdrant("Are You Weak",embedding_model)
print(result_translations)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def create_translation_prompt(query_text,related_translations):
    # Define the translation task with dynamic source and target languages
    # prompt = f"{query_text}"
    # return prompt
    """
    Create a dynamic translation prompt with related translations for context.
    The related translations are not to be translated, only the query text should be translated.
    """
    prompt = f"Translate the following text from English to French:\n{query_text}\n\n"
    
    # Add related translations for cultural context (these should not be translated)
    prompt += "here are some Related translations (for cultural context, not directly related to the query):\n"
    for translation in related_translations:
        prompt += f"- {translation}\n"
    
    prompt += "\nNow translate only the given text above into French, not the related translations."

    return prompt

def translate_with_llm(query_text, translations, max_length=512):

    prompt = create_translation_prompt(query_text,translations)
    print("My promt is")
    print(prompt)

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    # Ensure the model and inputs are on the same device
    device = model.device

    # Generate the response
    outputs = model.generate(
        inputs["input_ids"].to(device),  # Ensure that input is on the correct device
        max_length=max_length,  # Set the maximum length for the response
        temperature=0.7,  # Control randomness
        num_return_sequences=1,  # Get only one response
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn")
    )

    # Decode the generated text
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Function that combines the search and translation steps
def get_translation(query_text, embedding_model):
    # Step 1: Query Qdrant for related translations
    related_translations = query_qdrant(query_text, embedding_model)

    # Format related translations into a readable string for the Llama prompt
    formatted_translations = "\n".join(related_translations)

    # Step 2: Use the related translations and query to get a nuanced translation from the LLM
    translation = translate_with_llm(query_text, related_translations)
    

    

    return translation



df_test = pd.read_csv("test.csv")

# Initialize the output column
df_test['output'] = df_test['en'].apply(lambda x: get_translation(x, embedding_model))

df_test.to_excel("test_output.xlsx", index=False)


print(translated_text)





