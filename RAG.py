from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from LLM import LLM

class RAG(LLM):           
    def _create_translation_prompt(self, query_text, related_translations):
        prompt = f"[RAG Custom] Translate this text from English to French:\n{query_text}\n\n"
        prompt += "Here are some additional cultural notes (if relevant):\n"
        for translation in related_translations:
            prompt += f"- {translation}\n"
        prompt += "\nPlease provide the best French translation for the query text."
        return prompt
    
    def translate_with_llm(self, query_text, related_translations, max_length=512):
        prompt = self._create_translation_prompt(query_text, related_translations)
                
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        device = self.model.device
        outputs = self.model.generate(
            inputs["input_ids"].to(device),
            max_length=max_length,
            temperature=0.7,
            num_return_sequences=1,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("fra_Latn")
        )
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
