from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# AutoTokenizer.from_pretrained("facebook/opt-350m")

AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)