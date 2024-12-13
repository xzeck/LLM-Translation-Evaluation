#!/usr/bin/env python
# coding: utf-8


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


from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2-2b-it"
hf_token = ""

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model for causal language modeling (appropriate for Llama)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)



def create_translation_prompt(query_text, related_translations):
    prompt = (
        f"You are a highly skilled translator. Translate the following English text to French:\n"
        f"<<START>>{query_text}<<END>>\n\n"
        f"Contextual translations (do not translate, provided for reference):\n"
    )
    for translation in related_translations:
        prompt += f"- {translation}\n"
    prompt += "\nProvide only the French translation of the text between <<START>> and <<END>>."
    return prompt



def translate_with_llm(query_text, related_translations, max_length=512):
    """
    Translates the query_text to French using a model with related translations as context.
    """
    # Generate the translation prompt
    prompt = create_translation_prompt(query_text, related_translations)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    # Check GPU availability and move model and inputs to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the translation
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=0.7,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("<s><fr>")  # Force French output
    )


    # Decode the output
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation



# # Function that combines the search and translation steps
# def get_translation(query_text, embedding_model):
#     # Step 1: Query Qdrant for related translations
#     related_translations = query_qdrant(query_text, embedding_model)

#     # Format related translations into a readable string for the Llama prompt
#     formatted_translations = "\n".join(related_translations)

#     # Step 2: Use the related translations and query to get a nuanced translation from the LLM
#     translation = translate_with_llm(query_text, related_translations)
    

    

#     return translation
def get_translation(query_text, embedding_model):
    """
    Combines Qdrant search and LLM translation steps to provide a nuanced translation.
    """
    # Step 1: Query Qdrant for related translations
    related_translations = query_qdrant(query_text, embedding_model)

    # Step 2: Generate the prompt and call the LLM
    prompt = create_translation_prompt(query_text, related_translations)
    raw_translation = translate_with_llm(query_text, related_translations)

    # Extract the text between markers
    start_marker = "<<START>>"
    end_marker = "<<END>>"
    start_index = raw_translation.find(start_marker) + len(start_marker)
    end_index = raw_translation.find(end_marker)
    
    if start_index != -1 and end_index != -1:
        translated_text = raw_translation[start_index:end_index].strip()
    else:
        # Fallback: Use the entire raw_translation or a default message
        translated_text = raw_translation.strip()

    return translated_text






df_test = pd.read_csv("test.csv")

# Initialize the output column
df_test['output'] = df_test['en'].apply(lambda x: get_translation(x, embedding_model))

df_test.to_excel("test_output.xlsx", index=False)








import pandas as pd
import sacrebleu

# Load the Excel file
df_test = pd.read_excel("test_output.xlsx")

# Ensure the two columns are strings and prepare tokens
df_test['output_tokens'] = df_test['output'].apply(lambda x: str(x).strip())
df_test['fr_tokens'] = df_test['fr'].apply(lambda x: str(x).strip())

# Prepare references and hypotheses
references = [[ref] for ref in df_test['fr_tokens'].tolist()]  # List of lists of reference translations
hypotheses = df_test['output_tokens'].tolist()  # List of predicted translations

# Calculate BLEU score
bleu_score = sacrebleu.corpus_bleu(hypotheses, references)
print(f"BLEU Score: {bleu_score.score:.4f}")

