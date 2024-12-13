from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLM:
    def __init__(self, model_name, token):
        
        if model_name == None:
            raise ValueError("No translation model name given")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token)
        
    def __create_translation_prompt(self, query_text):
        prompt = f"Translate the following text from English to French:\n{query_text}\n\n"
        return prompt
        
    def translate_with_llm(self, query_text, max_length=512):
        prompt = self.__create_translation_prompt(query_text)
        
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
