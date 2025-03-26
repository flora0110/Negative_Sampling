from transformers import AutoModelForCausalLM, AutoTokenizer

AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
AutoTokenizer.from_pretrained("facebook/opt-350m")