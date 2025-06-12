# 实验配置

## Jaeger后端

- 注意：Jaeger后端需要在宿主机上启动，在容器环境中会启动失败（autoDL启动不了）
- 启动命令：
    ```
    docker run --rm --name jaeger \
        -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
        -p 6831:6831/udp \
        -p 6832:6832/udp \
        -p 5778:5778 \
        -p 16686:16686 \
        -p 4317:4317 \
        -p 4318:4318 \
        -p 14250:14250 \
        -p 14268:14268 \
        -p 14269:14269 \
        -p 9411:9411 \
        jaegertracing/all-in-one:1.57
    ```

## Client

- 环境变量：
    ```
    export JAEGER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    export OTEL_SERVICE_NAME="client-service"
    ```
- 脚本路径：`examples/online_serving/opentelemetry/dummy_client.py`

## Server

### 非PD分离部署模式

- 环境变量：
    ```
    export JAEGER_IP=$(docker inspect   --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    export OTEL_SERVICE_NAME="vllm-server"
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    ```
- launch.json:
    ```
    {
        "version": "0.2.0",
        "configurations": [
            {
            "name": "Debug vllm serve",
            "type": "debugpy",
            "request": "launch",
            "module": "vllm.entrypoints.openai.api_server",
            "args": [
                "--model", "facebook/opt-125m",
                "--otlp-traces-endpoint", "grpc://localhost:4317",
                "--max-model-len", "64",
                "--max-num-seqs", "1",
                "--gpu-memory-utilization", "0.05",
            ],
            "console": "integratedTerminal",
            "justMyCode": false
            }
        ]
    }
    ```

### PD分离部署模式

- 环境变量:
    ```
    export JAEGER_IP=$(docker inspect   --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    export OTEL_SERVICE_NAME="vllm-server"
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    export LMCACHE_CONFIG_FILE=/home/haonan/vllm/examples/others/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml
    export UCX_TLS=cuda_ipc,cuda_copy,tcp
    export LMCACHE_USE_EXPERIMENTAL=True
    export VLLM_ENABLE_V1_MULTIPROCESSING=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export CUDA_VISIBLE_DEVICES=0
    ```
- launch.json
    ```
    {
        "version": "0.2.0",
        "configurations": [
            {
            "name": "Debug vllm serve",
            "type": "debugpy",
            "request": "launch",
            "module": "vllm.entrypoints.openai.api_server",
            "args": [
                "--model", "facebook/opt-125m",
                "--otlp-traces-endpoint", "grpc://localhost:4317",
                "--max-model-len", "64",
                "--max-num-seqs", "1",
                "--gpu-memory-utilization", "0.05",
                "--disable-log-requests",
                "--enforce-eager",
                "--kv-transfer-config", "{\"kv_connector\":\"LMCacheConnector\",\"kv_role\":\"kv_producer\", "kv_connector_extra_config\": {\"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"producer1\"}}"   
            ],
            "console": "integratedTerminal",
            "justMyCode": false
            }
        ]
    }
    ```

