FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-3.py310
WORKDIR /

COPY environment.yml .
COPY requirements.txt .
COPY wandb_api_key.txt .

COPY src/ src/
COPY scripts/ scripts/


RUN pip install -r requirements.txt