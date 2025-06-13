from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

class LLMIntegration:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def query(self, input_text: str) -> str:
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=150)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def load_llm(model_name: str) -> LLMIntegration:
    return LLMIntegration(model_name)

def query_llm(prompt, model_name='facebook/bart-large-cnn'):
    """
    Queries the specified LLM model with a given prompt and returns the response.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response