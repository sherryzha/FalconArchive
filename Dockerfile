# Start with the specified base image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04

# Upgrade pip (optional but often necessary)
RUN pip install --upgrade pip

# Upgrade Transformers and install other necessary libraries
RUN pip install -U trl==0.7.1 \
    transformers==4.32.1 \
    accelerate==0.22.0 \
    peft==0.5.0 \
    datasets==2.14.5 \
    bitsandbytes==0.41.1 \
    einops==0.6.1 \
    wandb==0.15.10 \
    huggingface_hub==0.16.4 \
    numpy==1.26.0 \
    pandas>=2.0.0

# (Optional) Copy your model and any other necessary files into the image
# COPY ./my_model /opt/ml/model
ADD . /opt/program/

RUN ls /opt/program/

WORKDIR /opt/program

COPY ./script.py /opt/program/script.py
# Set huggingface token. IDK if I should hardcode it, but I put it here just in case.
RUN huggingface-cli login --token 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

# Set up the entry point for the inference code
ENTRYPOINT ["python", "/opt/program/script.py"]
