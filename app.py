from transformers import AutoTokenizer, AutoModelForCausalLM

class InferlessPythonModel:
    def initialize(self):
        model_id = "CohereForAI/c4ai-command-r-v01"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def infer(self,inputs):
        prompt = inputs["prompt"]
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        generated_tokens = self.model.generate(input_ids,max_new_tokens=256,do_sample=True,temperature=0.1)
        generated_text = self.tokenizer.batch_decode(generated_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        return { "generated_text" : generated_text}
        
    def finalize(self):
        self.pipe = None
