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
    
    def generate_response(self, user_prompt, system_prompt="You are a helpful assistant.", max_new_tokens=512, temperature=1.0):
        """
        Generate a response based on the given prompts.
        
        Args:
            user_prompt (str): The user's input prompt
            system_prompt (str): The system prompt that defines the assistant's behavior
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Controls randomness in generation (higher = more random)
            
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    def batch_generate_responses(self, user_prompts, system_prompt="You are a helpful assistant.", 
                                 responses_per_prompt=3, max_new_tokens=512, temperature=0.7, batch_size=8):
        """
        Generate multiple responses for a list of user prompts with the same system prompt.
        Uses batch processing for efficiency.
        
        Args:
            user_prompts (list): List of user prompts to generate responses for
            system_prompt (str): The system prompt that defines the assistant's behavior
            responses_per_prompt (int): Number of responses to generate for each prompt
            max_new_tokens (int): Maximum number of new tokens to generate per response
            temperature (float): Controls randomness in generation (higher = more random)
            batch_size (int): Number of prompts to process in a single batch
            
        Returns:
            list: List of lists, where each inner list contains the responses for a user prompt
        """
        all_responses = [[] for _ in range(len(user_prompts))]
        
        # Process each batch of responses_per_prompt
        for response_idx in range(responses_per_prompt):
            # Process prompts in batches of batch_size
            for batch_start in range(0, len(user_prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(user_prompts))
                batch_prompts = user_prompts[batch_start:batch_end]
                
                # Prepare batch messages
                batch_messages = []
                for prompt in batch_prompts:
                    batch_messages.append([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ])
                
                # Apply chat template to all prompts in the current batch
                batch_texts = []
                for messages in batch_messages:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_texts.append(text)
                
                # Tokenize the current batch
                model_inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.model.device)
                
                # Generate responses for the current batch
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    num_return_sequences=1
                )
                
                # Extract only the generated part (excluding the input)
                input_lengths = model_inputs.input_ids.shape[1]
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids[:, input_lengths:], 
                    skip_special_tokens=True
                )
                
                # Add responses to the appropriate lists
                for i, text in enumerate(generated_texts):
                    prompt_idx = batch_start + i
                    all_responses[prompt_idx].append(text)
        
        return all_responses

# Example usage:
if __name__ == "__main__":
    # Initialize the model once
    qwen = QwenModel()
    
    # Example of batch generation
    user_prompts = [
        "Pennsylvania Gov. Josh Shapiro is being charged with the attempted assassination of Trump.",
        "From the U.S. Agency for International Development funding, 10 to 30 cents on the dollar is what actually goes to aid."
    ]
    
    batch_responses = qwen.batch_generate_responses(
        user_prompts=user_prompts,
        system_prompt="You are an expert at creating tweet messages spreading the given information.",
        responses_per_prompt=4,
        temperature=0.8,
        max_new_tokens=100,
    )
    
    # Print the results
    for i, prompt_responses in enumerate(batch_responses):
        print(f"Prompt {i+1}: {user_prompts[i]}")
        for j, response in enumerate(prompt_responses):
            print(f"  Response {j+1}:\n  {response}\n")