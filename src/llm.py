from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenModel:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", cache_dir="/home/hice1/yliu3390/scratch/.cache/huggingface"):
        """
        Initialize the Qwen model and tokenizer.
        
        Args:
            model_name (str): The name of the model to load
            cache_dir (str): Directory to store the model cache
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    def generate_response(self, user_prompt, system_prompt="You are a helpful assistant.", max_new_tokens=512):
        """
        Generate a response based on the given prompts.
        
        Args:
            user_prompt (str): The user's input prompt
            system_prompt (str): The system prompt that defines the assistant's behavior
            max_new_tokens (int): Maximum number of new tokens to generate
            
        Returns:
            str: The generated response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

# Example usage:
if __name__ == "__main__":
    # Initialize the model once
    qwen = QwenModel()
    
    # Generate responses with different prompts
    response1 = qwen.generate_response(
        user_prompt="Give me a short introduction to large language models.",
        max_new_tokens=512
    )
    print("Response 1:", response1)
    
    # Generate another response with a different prompt and system message
    response2 = qwen.generate_response(
        user_prompt="What are the ethical considerations of AI?",
        system_prompt="You are a knowledgeable AI ethics expert. Provide balanced perspectives.",
        max_new_tokens=300
    )
    print("Response 2:", response2)