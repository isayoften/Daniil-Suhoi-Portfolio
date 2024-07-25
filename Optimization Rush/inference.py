from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

tokenizer = AutoTokenizer.from_pretrained(config.model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(config.model_id, quantization_config=config.bnb_config)
lora_model = PeftModel.from_pretrained(model, "model")

prompt = 'The great chef shared a recipe for a very tasty dish. Ingredients: tomatoes, mushrooms, eggs, salt. Cooking instructions:\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

outputs = model.generate(**inputs, num_beams=5, max_new_tokens=300)
generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated[0])

