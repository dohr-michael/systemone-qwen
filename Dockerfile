# systemone-qwen: one static Go binary on an empty image. The GPU lives in llama-server.
# The build stage runs natively and cross-compiles: linux/amd64 and linux/arm64 need no emulation.
ARG GO_IMAGE=golang:1.27.3

FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS build
ARG TARGETOS TARGETARCH
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w" -o /out/systemone-qwen ./cmd/systemone-qwen

FROM scratch
# Root certificates, for an HTTPS backend (LiteLLM behind TLS).
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=build /out/systemone-qwen /systemone-qwen
ENV SYSTEMONE_HOST=0.0.0.0 SYSTEMONE_PORT=8000
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/systemone-qwen"]
