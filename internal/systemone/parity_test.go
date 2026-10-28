package systemone

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Replays testdata/parity.json, recorded from the benchmarked Python service by
// testdata/generate_parity.py: same prompts and backend requests, same answers.

type parityCase struct {
	Name          string          `json:"name"`
	Config        map[string]any  `json:"config"`
	BackendStatus int             `json:"backend_status"`
	Request       string          `json:"request"`
	Status        int             `json:"status"`
	Response      json.RawMessage `json:"response"`
	Calls         []any           `json:"calls"`
}

type roundTripper func(*http.Request) (*http.Response, error)

func (f roundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func reply(status int, body any) (*http.Response, error) {
	recorder := httptest.NewRecorder()
	if text, ok := body.(string); ok {
		recorder.WriteHeader(status)
		recorder.WriteString(text)
	} else {
		writeJSON(recorder, status, body)
	}
	return recorder.Result(), nil
}

// parityFake mirrors generate_parity.py's Fake.
type parityFake struct {
	status int
	mu     sync.Mutex
	calls  []string // canonical JSON of [path, body]
}

func canonical(v any) string {
	data, _ := json.Marshal(v)
	return string(data)
}

func (f *parityFake) handle(r *http.Request) (*http.Response, error) {
	var body map[string]any
	if data := readBody(r); len(data) > 0 {
		if err := json.Unmarshal(data, &body); err != nil {
			return nil, err
		}
	}
	if r.URL.Path != "/health" {
		f.mu.Lock()
		f.calls = append(f.calls, canonical([]any{r.URL.Path, body}))
		f.mu.Unlock()
	}
	if f.status != http.StatusOK {
		return reply(f.status, "boom")
	}
	if maxTokens, _ := body["max_tokens"].(float64); maxTokens > 1 {
		return reply(200, map[string]any{"choices": []any{map[string]any{"text": "  It depends on the receipt.\n  "}},
			"usage": map[string]any{"prompt_tokens": 50, "completion_tokens": 7}})
	}
	prompt := body["prompt"].(string)
	type entry struct {
		token   string
		logprob float64
	}
	var entries []entry
	for _, letter := range alphabet {
		h := sha256.Sum256([]byte(prompt + "\x00" + string(letter)))
		if h[4]%5 != 0 {
			entries = append(entries, entry{string(letter), -8 * float64(binary.BigEndian.Uint32(h[:4])) / (1 << 32)})
		}
	}
	for filler := 0; len(entries) < int(body["logprobs"].(float64)); filler++ {
		entries = append(entries, entry{"~" + strconv.Itoa(filler), -20 - float64(filler)/100})
	}
	top := make([]any, len(entries))
	for i, e := range entries {
		top[i] = map[string]any{"id": 1000 + i, "token": e.token, "logprob": e.logprob}
	}
	return reply(200, map[string]any{
		"choices": []any{map[string]any{"text": entries[0].token, "logprobs": map[string]any{"content": []any{
			map[string]any{"id": 1000, "token": entries[0].token, "logprob": entries[0].logprob, "top_logprobs": top}}}}},
		"usage": map[string]any{"prompt_tokens": 40, "completion_tokens": 1}})
}

// sameValue compares in key order; numbers within float rounding.
func sameValue(path string, got, want Value) error {
	if got.Kind != want.Kind {
		return fmt.Errorf("%s: kind %d, want %d", path, got.Kind, want.Kind)
	}
	switch got.Kind {
	case Number:
		a, _ := strconv.ParseFloat(got.Text, 64)
		b, _ := strconv.ParseFloat(want.Text, 64)
		if math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(b)) {
			return fmt.Errorf("%s: %s, want %s", path, got.Text, want.Text)
		}
	case Array:
		if len(got.Items) != len(want.Items) {
			return fmt.Errorf("%s: %d items, want %d", path, len(got.Items), len(want.Items))
		}
		for i := range got.Items {
			if err := sameValue(fmt.Sprintf("%s[%d]", path, i), got.Items[i], want.Items[i]); err != nil {
				return err
			}
		}
	case Object:
		if len(got.Members) != len(want.Members) {
			return fmt.Errorf("%s: %d keys, want %d", path, len(got.Members), len(want.Members))
		}
		for i := range got.Members {
			if got.Members[i].Key != want.Members[i].Key {
				return fmt.Errorf("%s: key %q at %d, want %q", path, got.Members[i].Key, i, want.Members[i].Key)
			}
			if err := sameValue(path+"."+got.Members[i].Key, got.Members[i].Value, want.Members[i].Value); err != nil {
				return err
			}
		}
	default:
		if got.Text != want.Text || got.Bool != want.Bool {
			return fmt.Errorf("%s: %q, want %q", path, got.Text, want.Text)
		}
	}
	return nil
}

func TestParityWithThePythonService(t *testing.T) {
	data, err := os.ReadFile("testdata/parity.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []parityCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			model := map[string]any{"base_url": "http://llama"}
			for k, v := range c.Config {
				model[k] = v
			}
			settings, err := ParseSettings([]byte(canonical(map[string]any{
				"models": map[string]any{"qwen-decision": model}, "aliases": map[string]any{"jev-latest": "qwen-decision"}})))
			if err != nil {
				t.Fatal(err)
			}
			fake := &parityFake{status: c.BackendStatus}
			backends := map[string]*Backend{"qwen-decision": NewBackend(settings.Models[0].Config, roundTripper(fake.handle))}
			server := httptest.NewServer(NewHandler(settings, backends, slog.New(slog.DiscardHandler)))
			defer server.Close()

			response, err := http.Post(server.URL+"/api/v1/systemone", "application/json", strings.NewReader(c.Request))
			if err != nil {
				t.Fatal(err)
			}
			body, _ := io.ReadAll(response.Body)
			response.Body.Close()
			if response.StatusCode != c.Status {
				t.Fatalf("status %d, want %d: %s", response.StatusCode, c.Status, body)
			}
			if c.Status != http.StatusOK {
				return
			}

			want := make([]string, len(c.Calls))
			for i, call := range c.Calls {
				want[i] = canonical(call)
			}
			got := slices.Clone(fake.calls)
			slices.Sort(got)
			slices.Sort(want)
			if !slices.Equal(got, want) {
				t.Fatalf("backend requests differ:\ngot  %v\nwant %v", got, want)
			}

			gotValue, err := ParseJSON(body)
			if err != nil {
				t.Fatal(err)
			}
			gotValue.Members = slices.DeleteFunc(gotValue.Members, func(m Member) bool { return m.Key == "id" })
			wantValue, _ := ParseJSON(c.Response)
			if err := sameValue("response", gotValue, wantValue); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// readBody: client-side requests without a body carry a nil Body.
func readBody(r *http.Request) []byte {
	if r.Body == nil {
		return nil
	}
	data, _ := io.ReadAll(r.Body)
	return data
}
