from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import torch

tokenizer = AutoTokenizer.from_pretrained(config.model_id)
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

model = AutoModelForCausalLM.from_pretrained(config.model_id, 
                                             quantization_config=config.bnb_config,  
                                             device_map='auto', 
                                             torch_dtype=torch.bfloat16)
model.config.pad_token_id = tokenizer.pad_token_id  

lora_model = PeftModel.from_pretrained(model, "model")

prompt = "One wise man said: “When a woman is angry "
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

outputs = lora_model.generate(**inputs, do_sample = True, top_p = 0.8, max_new_tokens=300)
generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated[0])

