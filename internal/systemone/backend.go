package systemone

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Client for one llama-server: OpenAI /v1/completions with logprobs, plus /tokenize.
//
// Logprobs come back in two shapes: llama-server's `logprobs.content[0].top_logprobs` (a list
// carrying token ids) and the legacy OpenAI/vLLM `logprobs.top_logprobs[0]` (a token -> logprob
// map). Both are read. llama-server reports raw logprobs (before temperature), so the
// requested sampling temperature does not change them.

// BackendError: the model server failed or answered something unusable (HTTP 502 to our caller).
type BackendError struct{ msg string }

func (e *BackendError) Error() string { return e.msg }

func backendErrorf(format string, args ...any) error {
	return &BackendError{fmt.Sprintf(format, args...)}
}

type alternative struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	ID      *int    `json:"id"`
}

type token struct {
	ID    int
	Piece string
}

type Usage struct {
	InputTokens, OutputTokens atomic.Int64
}

type completion struct {
	Choices []struct {
		Text     string `json:"text"`
		Logprobs *struct {
			Content []struct {
				TopLogprobs []alternative `json:"top_logprobs"`
			} `json:"content"`
			TopLogprobs []map[string]float64 `json:"top_logprobs"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
}

func (c *completion) firstTokenAlternatives() ([]alternative, error) {
	if len(c.Choices) == 0 || c.Choices[0].Logprobs == nil {
		return nil, backendErrorf("completion without logprobs; the backend must support `logprobs`")
	}
	logprobs := c.Choices[0].Logprobs
	var alternatives []alternative
	if len(logprobs.Content) > 0 {
		alternatives = logprobs.Content[0].TopLogprobs
	} else if len(logprobs.TopLogprobs) > 0 {
		for tok, logprob := range logprobs.TopLogprobs[0] {
			alternatives = append(alternatives, alternative{Token: tok, Logprob: logprob})
		}
	}
	if len(alternatives) == 0 {
		return nil, backendErrorf("completion returned no top logprobs")
	}
	return alternatives, nil
}

type Backend struct {
	Config ModelConfig
	client *http.Client
	base   string
	apiKey string
	slots  chan struct{}

	mu     sync.Mutex
	tokens map[string][]token
}

// NewBackend: a nil transport means http.DefaultTransport.
func NewBackend(cfg ModelConfig, transport http.RoundTripper) *Backend {
	apiKey := ""
	if cfg.APIKeyEnv != "" {
		apiKey = os.Getenv(cfg.APIKeyEnv)
	}
	return &Backend{
		Config: cfg,
		client: &http.Client{Transport: transport, Timeout: time.Duration(cfg.TimeoutS * float64(time.Second))},
		base:   strings.TrimRight(cfg.BaseURL, "/"),
		apiKey: apiKey,
		slots:  make(chan struct{}, cfg.Concurrency),
		tokens: map[string][]token{},
	}
}

func (b *Backend) post(ctx context.Context, path string, body, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, b.base+path, bytes.NewReader(payload))
	if err != nil {
		return backendErrorf("%s: %v", path, err)
	}
	request.Header.Set("Content-Type", "application/json")
	if b.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+b.apiKey)
	}
	select {
	case b.slots <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}
	response, err := b.client.Do(request)
	if err != nil {
		<-b.slots
		if ctx.Err() != nil {
			return ctx.Err() // our caller left, or a sibling question failed
		}
		return backendErrorf("%s: %v", path, err)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, 64<<20))
	response.Body.Close()
	<-b.slots
	if err != nil {
		return backendErrorf("%s: %v", path, err)
	}
	if response.StatusCode != http.StatusOK {
		return backendErrorf("%s returned %d: %.300s", path, response.StatusCode, data)
	}
	if err := json.Unmarshal(data, out); err != nil {
		return backendErrorf("%s: unreadable answer: %v", path, err)
	}
	return nil
}

func (b *Backend) completion(ctx context.Context, prompt string, usage *Usage, fields map[string]any) (*completion, error) {
	body := maps.Clone(b.Config.ExtraBody)
	if body == nil {
		body = map[string]any{}
	}
	body["prompt"] = prompt
	maps.Copy(body, fields)
	if b.Config.ServedModel != "" {
		body["model"] = b.Config.ServedModel
	}
	var out completion
	if err := b.post(ctx, "/v1/completions", body, &out); err != nil {
		return nil, err
	}
	usage.InputTokens.Add(out.Usage.PromptTokens)
	usage.OutputTokens.Add(out.Usage.CompletionTokens)
	return &out, nil
}

func (b *Backend) firstToken(ctx context.Context, prompt string, usage *Usage, top int) ([]alternative, error) {
	out, err := b.completion(ctx, prompt, usage, map[string]any{"max_tokens": 1, "temperature": 0, "logprobs": top})
	if err != nil {
		return nil, err
	}
	return out.firstTokenAlternatives()
}

// think returns reasoning text, with Qwen's recommended thinking sampler, cut at </think> or the budget.
func (b *Backend) think(ctx context.Context, prompt string, usage *Usage) (string, error) {
	out, err := b.completion(ctx, prompt, usage, map[string]any{
		"max_tokens": b.Config.ThinkTokens, "temperature": 0.6, "top_p": 0.95, "seed": 0, "stop": []string{"</think>"},
	})
	if err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", nil
	}
	return out.Choices[0].Text, nil
}

// tokenize calls llama-server /tokenize -> [(id, piece)]; only reachable with direct server access.
func (b *Backend) tokenize(ctx context.Context, text string) ([]token, error) {
	b.mu.Lock()
	cached, ok := b.tokens[text]
	b.mu.Unlock()
	if ok {
		return cached, nil
	}
	var out struct {
		Tokens []json.RawMessage `json:"tokens"`
	}
	body := map[string]any{"content": text, "add_special": false, "with_pieces": true}
	if err := b.post(ctx, "/tokenize", body, &out); err != nil {
		return nil, err
	}
	tokens := make([]token, len(out.Tokens))
	for i, raw := range out.Tokens {
		var withPiece struct {
			ID    int    `json:"id"`
			Piece string `json:"piece"`
		}
		if err := json.Unmarshal(raw, &withPiece); err != nil {
			if err := json.Unmarshal(raw, &withPiece.ID); err != nil {
				return nil, backendErrorf("/tokenize: unreadable token %s", raw)
			}
		}
		tokens[i] = token{withPiece.ID, withPiece.Piece}
	}
	b.mu.Lock()
	b.tokens[text] = tokens
	b.mu.Unlock()
	return tokens, nil
}

func (b *Backend) Healthy(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, b.base+"/health", nil)
	if err != nil {
		return false
	}
	response, err := b.client.Do(request)
	if err != nil {
		return false
	}
	response.Body.Close()
	return response.StatusCode == http.StatusOK
}
