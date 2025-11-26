FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .

# Install mim + mmengine + mmcv
RUN pip install --no-cache-dir openmim && \
    mim install mmengine && \
    mim install "mmcv==2.1.0"

# Install opencv-python-headless explicitly
RUN pip install --no-cache-dir opencv-python-headless

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
