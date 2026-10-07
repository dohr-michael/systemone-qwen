# Deployment

```
client ──► LiteLLM pass-through (optional) ──► systemone-qwen ──► llama-server (GPU)
           /decision/api/v1/systemone          :8000              :8080, Qwen GGUF
```

systemone-qwen holds no weights and needs no GPU, CUDA or ROCm: it builds prompts, calls
llama-server over HTTP and turns logprobs into answers. The image is multi-arch
(`linux/amd64`, `linux/arm64`). Any llama-server build works (ROCm, CUDA, Vulkan, CPU), for
example [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) on
Strix Halo.

## llama-server for decisions

A dedicated instance, separate from chat, keeps decisions from queueing behind long generations:

```sh
llama-server --model /models/Qwen3.5-9B-Q8_0.gguf --host 0.0.0.0 --port 8080 \
  --gpu-layers 999 --flash-attn 1 --ctx-size 16384 --parallel 1 \
  --cache-type-k f16 --cache-type-v f16 --no-mmap --metrics --no-ui
```

- `--parallel 1`: reproducible probabilities. llama.cpp logits can shift slightly with batch
  composition. Raise it (and `concurrency` in the config) only if throughput matters more.
  The context is split across slots.
- `--cache-type-k/v f16`: the context is small, and a quantized KV cache can shift logits.
- `--ctx-size 16384`: JevBench's longest prompts fit; decisions need far less than chat.
- No `--mmproj`, no `--embeddings`, no chat template flags: the service sends a raw prompt
  with thinking already closed.

Check it from inside the cluster before wiring the service:

```sh
curl -s http://<llama>:8080/tokenize -d '{"content":"<|im_start|>A","with_pieces":true}'   # <|im_start|> is ONE token
```

## Kubernetes

Co-locate the service with its llama-server through pod affinity (the service needs no GPU,
only proximity), and talk to the llama-server Service directly: that skips a proxy hop and
keeps `/tokenize` available.

```yaml
apiVersion: v1
kind: ConfigMap
metadata: {name: systemone-qwen, namespace: ai-stack}
data:
  config.yaml: |
    aliases: {jev-latest: qwen-decision}
    models:
      qwen-decision:
        base_url: http://qwen-decision-llm-server.ai-stack.svc.cluster.local:8080
        calibration_temperature: 1.5
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: systemone-qwen, namespace: ai-stack}
spec:
  replicas: 1
  selector: {matchLabels: {app: systemone-qwen}}
  template:
    metadata: {labels: {app: systemone-qwen}}
    spec:
      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector: {matchLabels: {app: qwen-decision}}   # the llama-server pod's label
              topologyKey: kubernetes.io/hostname
      tolerations:
        - {key: dedicated, operator: Equal, value: ai, effect: NoSchedule}
      containers:
        - name: systemone-qwen
          image: ghcr.io/dohr-michael/systemone-qwen:latest
          env:
            - {name: SYSTEMONE_CONFIG, value: /config/config.yaml}
          ports: [{name: http, containerPort: 8000}]
          readinessProbe: {httpGet: {path: /ready, port: http}, periodSeconds: 10}
          livenessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 30}
          resources: {requests: {cpu: 100m, memory: 128Mi}, limits: {memory: 512Mi}}
          volumeMounts: [{name: config, mountPath: /config, readOnly: true}]
      volumes:
        - {name: config, configMap: {name: systemone-qwen}}
---
apiVersion: v1
kind: Service
metadata: {name: systemone-qwen, namespace: ai-stack}
spec:
  selector: {app: systemone-qwen}
  ports: [{name: http, port: 8000, targetPort: http}]
```

## Exposing it through LiteLLM

LiteLLM's built-in `/typesafe/*` route forwards to TypeSafe itself. A generic pass-through on a
distinct path exposes this service with LiteLLM keys, budgets and logs:

```yaml
general_settings:
  pass_through_endpoints:
    - path: "/decision/api/v1/systemone"
      target: "http://systemone-qwen.ai-stack.svc.cluster.local:8000/api/v1/systemone"
      auth: true
```

Clients then use the base URL `https://<litellm>/decision/api/v1`, for instance OpenRouter's
SDK with `server_url` overridden.
