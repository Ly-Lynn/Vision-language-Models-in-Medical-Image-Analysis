from transformers import CLIPModel, AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("local_model/clinical_bert", 
                                             use_safetensors=False, 
                                             local_files_only=True,
                                             trust_remote_code=True)

print(model)