FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

# Upgrade PyTorch to 2.4 inside container
RUN pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install compatible transformers version
RUN pip install \
    transformers==4.41.2 \
    datasets \
    evaluate \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    huggingface_hub \
    accelerate

CMD ["bash"]
