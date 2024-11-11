from transformers import AutoModelForCausalLM , AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch
from peft import PeftModel
import string
import re
from transformers import pipeline
import io
import json
import pandas as pd

def model_fn(model_dir):
    model_name = "tiiuae/falcon-7b"
        
    #Set configuration for bits & bytes & load in base Falcon Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False   
    
    model = PeftModel.from_pretrained(model, model_dir+"/falcon-rewrite-560")
    model.config.pad_token_id = model.config.eos_token_id
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    
    return {"model": model, "tokenizer": tokenizer}

def valid_start(input_string):
    '''
   Helper function for post_process.
   Input: Raw string.
   Output: String that begins at first letter or number.
    '''
    lead_char = input_string[0]
    if lead_char.isalpha() or lead_char.isdigit():       
        return lead_char.upper() + input_string[1:]
    else:
        return valid_start(input_string[1:])

def cut_off(input_string):
    '''
    Helper function for post_process.
    Input: Raw generated text.
    Output: Cleaned generated text that stops at last punctuation.
    '''
    punctuation_characters = ".?!"
    
    # Find the index of the last punctuation character
    last_punctuation_index = max(
        [input_string.rfind(p) for p in punctuation_characters]
    )

    # If no punctuation is found, return the original string
    if last_punctuation_index == -1:
        return input_string

    # Slice the string to the last punctuation (inclusive)
    cut_off_string = input_string[: last_punctuation_index + 1]

    return valid_start(cut_off_string)

def post_process(input_string):
    '''
    Post-processing function for cleaning generated text.
    Input: Raw generated text.
    Output: Cleaned generated text
    '''
    accepted_chars = r"[^a-zA-Z0-9.,!?/' ]"
    return re.sub(accepted_chars,'',cut_off(input_string.split("### Assistant: ")[1].replace("\n","").split("#")[0].split("Rewrite the following text ")[0])).strip()

'''
def input_fn(data, content_type):    
    #if content_type == "text/csv":
    #    # Parse CSV data and convert to appropriate format
    #    input_data = pd.read_csv(io.StringIO(data))
    #elif content_type == "application/json":
    #    # Parse JSON data and convert to appropriate format
    #    input_data = pd.read_json(data, typ = "series").to_list()
    #else:
    #    # Raise an error for unsupported content types
    #    raise ValueError(f"Unexpected Content type: {content_type}")
    return "###Human: " + data + "### Assistant: ".replace("\n","")

def predict_fn(data, model):
    out_text = model(data,
                    max_new_tokens=len(data.split(" ")),
                    do_sample=True,
                    temperature=1.2
                    )
    return out_text

def output_fn(prediction, accept):
    return post_process(prediction[0]['generated_text'])
'''
def transform_fn(model, input_data, content_type, accept):
    print(input_data)
    if content_type == "application/json":
        # Parse JSON data and convert to appropriate format
        input_data = pd.read_json(input_data, typ = "series")
    else:
        # Raise an error for unsupported content types
        raise ValueError(f"Unexpected Content type: {content_type}")
    generator = pipeline(input_data["parameters"]["task"], #"text2text-generation"
                     model=model["model"], 
                     tokenizer=model["tokenizer"],
                     use_cache=False,
                     torch_dtype=torch.bfloat16, 
                     device_map="auto"
                    )
    data = "###Human: " + input_data["input"] + "### Assistant: ".replace("\n","")
    out_text = generator(data,
                    max_new_tokens=len(data.split(" ")),
                    do_sample=True,
                    temperature=1.2
                    )
    cleaned_output = post_process(out_text[0]['generated_text'])
    return [{"generated_text": cleaned_output}]