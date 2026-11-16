#!/usr/bin/env bash
# Start HydraDB (single node, local disk) in the background, then the Streamlit demo on :7860.
set -e
cd "$(dirname "$0")"
mkdir -p data/prepared results /tmp/hydra/store /tmp/hydra/cache
for f in data/prepared/*.gz; do [ -e "$f" ] && gunzip -kf "$f"; done
printf 'space-demo-token-0123456789-0123456789' > /tmp/hydra/token

export RUST_MIN_STACK=33554432 GRAPH_USAGE_REPORT_ENABLED=false
export CLOUD_PROVIDER=local LOCAL_PATH=/tmp/hydra/store
export GRAPH_NAMESPACE=default GRAPH_ID=default GRAPH_CELL_ID=cell-0 GRAPH_CELLS=cell-0 GRAPH_NODE_ID=node-0
export GRAPH_DATA_PATH=data GRAPH_ALLOW_PLAINTEXT=true GRAPH_AUTH_TOKEN_FILE=/tmp/hydra/token
export GRAPH_DATA_CACHE_DIR=/tmp/hydra/cache GRAPH_DATA_CACHE_BYTES=2147483648
export GRAPH_BOLT_ADDR=127.0.0.1:7687 GRAPH_HTTP_ADDR=127.0.0.1:8443 GRAPH_ADMIN_ADDR=127.0.0.1:9090
export GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687

graph-node > /tmp/hydra/node.log 2>&1 &

export HYDRA_URL=http://127.0.0.1:8443 HYDRA_TOKEN="$(cat /tmp/hydra/token)" HYDRA_CELL=cell-0 HYDRA_NAMESPACE=default
exec streamlit run demo/app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true \
     --browser.gatherUsageStats false
