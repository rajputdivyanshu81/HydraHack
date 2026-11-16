# Hugging Face Space (Docker SDK, free CPU tier): HydraDB 0.4.0 + the DesirePath demo in one container.
# The HydraDB image is Ubuntu 24.04 without Python, so take its binary into an Ubuntu base that has both.
FROM ghcr.io/hydra-db/hydradb:0.4.0 AS hydra

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgraphblas7 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=hydra /usr/local/bin/graph-node /usr/local/bin/graph-node

# Spaces run the container as uid 1000
RUN userdel -r ubuntu 2>/dev/null || true; useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
RUN python3 -m venv /home/user/venv \
 && pip install --no-cache-dir streamlit httpx anthropic numpy python-dotenv

COPY --chown=user . /home/user/app
EXPOSE 7860
CMD ["bash", "start.sh"]
