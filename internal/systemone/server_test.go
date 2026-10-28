package systemone

import (
	"encoding/json"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// API behaviour against a fake llama-server (an http.RoundTripper): no model, no network.

// fakeLlama answers /v1/completions with chosen first-token logprobs and records every request.
// `table` maps a prompt to code -> logprob; codes missing from it are left out of the top-N,
// which is padded with unrelated tokens to the requested size, as a real llama-server does,
// unless `short` is set.
type fakeLlama struct {
	table   func(prompt string) map[string]float64
	legacy  bool
	short   bool
	status  int
	noIDs   bool
	tokens  bool // serve /tokenize
	mu      sync.Mutex
	request []map[string]any
	paths   []string
}

func newFake(table map[string]float64) *fakeLlama {
	if table == nil {
		table = map[string]float64{"A": -0.1, "B": -2.5}
	}
	return &fakeLlama{table: func(string) map[string]float64 { return table }, status: 200, tokens: true}
}

func tokenID(piece string) int {
	id := 1000
	for _, r := range piece {
		id = id*31 + int(r)
	}
	return id
}

func (f *fakeLlama) handle(r *http.Request) (*http.Response, error) {
	var body map[string]any
	if data := readBody(r); len(data) > 0 {
		_ = json.Unmarshal(data, &body)
	}
	f.mu.Lock()
	f.paths = append(f.paths, r.URL.Path)
	if r.URL.Path == "/v1/completions" {
		f.request = append(f.request, body)
	}
	f.mu.Unlock()
	switch {
	case f.status != 200:
		return reply(f.status, "boom")
	case r.URL.Path == "/health":
		return reply(200, map[string]any{"status": "ok"})
	case r.URL.Path == "/tokenize" && !f.tokens:
		return reply(404, "not found")
	case r.URL.Path == "/tokenize":
		// One token per letter.
		var tokens []any
		for _, c := range body["content"].(string) {
			tokens = append(tokens, map[string]any{"id": tokenID(string(c)), "piece": string(c)})
		}
		return reply(200, map[string]any{"tokens": tokens})
	}
	if body["max_tokens"].(float64) > 1 { // thinking pass
		return reply(200, map[string]any{"choices": []any{map[string]any{"text": "It depends on the receipt."}},
			"usage": map[string]any{"prompt_tokens": 50, "completion_tokens": 7}})
	}
	table := f.table(body["prompt"].(string))
	var top []any
	legacy := map[string]float64{}
	add := func(tok string, logprob float64) {
		entry := map[string]any{"token": tok, "logprob": logprob}
		if !f.noIDs {
			entry["id"] = tokenID(tok)
		}
		top = append(top, entry)
		legacy[tok] = logprob
	}
	for _, letter := range alphabet {
		if logprob, ok := table[string(letter)]; ok {
			add(string(letter), logprob)
		}
	}
	for i := 0; !f.short && len(top) < int(body["logprobs"].(float64)); i++ {
		add("~"+strings.Repeat("x", i+1), -30)
	}
	logprobs := map[string]any{"content": []any{map[string]any{"token": "A", "logprob": 0, "top_logprobs": top}}}
	if f.legacy {
		logprobs = map[string]any{"top_logprobs": []any{legacy}, "content": nil}
	}
	return reply(200, map[string]any{"choices": []any{map[string]any{"text": "A", "logprobs": logprobs}},
		"usage": map[string]any{"prompt_tokens": 40, "completion_tokens": 1}})
}

func (f *fakeLlama) prompt(i int) string { return f.request[i]["prompt"].(string) }

type client struct {
	t   *testing.T
	url string
}

func serve(t *testing.T, fake *fakeLlama, configure func(*ModelConfig)) client {
	t.Helper()
	cfg := defaultModelConfig()
	cfg.BaseURL = "http://llama"
	if configure != nil {
		configure(&cfg)
	}
	settings := &Settings{Models: []NamedModel{{"qwen-decision", cfg}}, Aliases: []Alias{{"jev-latest", "qwen-decision"}}}
	if err := settings.validate(); err != nil {
		t.Fatal(err)
	}
	backends := map[string]*Backend{"qwen-decision": NewBackend(settings.Models[0].Config, roundTripper(fake.handle))}
	server := httptest.NewServer(NewHandler(settings, backends, slog.New(slog.DiscardHandler)))
	t.Cleanup(server.Close)
	return client{t, server.URL}
}

func (c client) do(method, path string, body any) (int, map[string]any) {
	c.t.Helper()
	var reader io.Reader
	if body != nil {
		data, _ := json.Marshal(body)
		reader = strings.NewReader(string(data))
	}
	request, _ := http.NewRequest(method, c.url+path, reader)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		c.t.Fatal(err)
	}
	defer response.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(response.Body).Decode(&out)
	return response.StatusCode, out
}

func (c client) ask(questions map[string]any, extra ...map[string]any) (int, map[string]any) {
	body := map[string]any{"model": "qwen-decision", "state": "I was charged twice.", "questions": questions}
	for _, e := range extra {
		for k, v := range e {
			body[k] = v
		}
	}
	return c.do(http.MethodPost, "/api/v1/systemone", body)
}

func (c client) answer(key string, question map[string]any) map[string]any {
	c.t.Helper()
	status, body := c.ask(map[string]any{key: question})
	if status != 200 {
		c.t.Fatalf("status %d: %v", status, body)
	}
	return body["answers"].(map[string]any)[key].(map[string]any)
}

var noul = map[string]any{"type": "noul", "instructions": "Does the customer ask for a refund?",
	"criteria": map[string]any{"true": "Asks for money back", "false": "Does not"}}

func softmax2(a, b float64) float64 { return math.Exp(a) / (math.Exp(a) + math.Exp(b)) }

func near(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestNoulReturnsTheProbabilityOfTheYesOption(t *testing.T) {
	fake := newFake(nil)
	status, body := serve(t, fake, nil).ask(map[string]any{"refund": noul})
	if status != 200 {
		t.Fatal(status, body)
	}
	near(t, body["answers"].(map[string]any)["refund"].(map[string]any)["noul"].(float64), softmax2(-0.1, -2.5))
	if body["model"] != "qwen-decision" || body["provider"] != "systemone-qwen" ||
		!strings.HasPrefix(body["id"].(string), "gen-dec-") {
		t.Fatal(body)
	}
	if usage := body["usage"].(map[string]any); usage["input_tokens"] != 40.0 || usage["output_tokens"] != 1.0 || usage["cost"] != 0.0 {
		t.Fatal(usage)
	}
	if !strings.Contains(fake.prompt(0), "A. Yes: Asks for money back\nB. No: Does not") ||
		fake.request[0]["cache_prompt"] != true || fake.request[0]["max_tokens"] != 1.0 {
		t.Fatal(fake.request[0])
	}
}

func TestChoiceShowsKeysAndReportsConfidence(t *testing.T) {
	fake := newFake(map[string]float64{"A": -0.2, "B": -1.9, "C": -4.0})
	answer := serve(t, fake, nil).answer("team", map[string]any{"type": "choice", "instructions": "Which team?",
		"criteria": map[string]any{"billing": "Payments, refunds", "technical": "Bugs", "sales": nil}})
	p := answer["probabilities"].(map[string]any)
	if answer["choice"] != "billing" {
		t.Fatal(answer)
	}
	near(t, p["billing"].(float64)+p["technical"].(float64)+p["sales"].(float64), 1)
	near(t, answer["confidence"].(float64), (3*p["billing"].(float64)-1)/2)
	// Go maps sort keys: the prompt lists them as the request did (here alphabetical).
	if !strings.Contains(fake.prompt(0), "A. billing: Payments, refunds\nB. sales\nC. technical: Bugs") {
		t.Fatal(fake.prompt(0))
	}
}

func TestScoreIsTheExpectedLevelWithLegend(t *testing.T) {
	answer := serve(t, newFake(map[string]float64{"A": -5, "B": -1, "C": -0.5}), nil).answer("urgency",
		map[string]any{"type": "score", "instructions": "How urgent?", "criteria": []any{"not urgent", "soon", "critical"}})
	p := answer["probabilities"].(map[string]any)
	near(t, answer["score"].(float64), p["1"].(float64)+2*p["2"].(float64))
	legend := answer["legend"].(map[string]any)
	if legend["0"] != "not urgent" || legend["1"] != "soon" || legend["2"] != "critical" {
		t.Fatal(legend)
	}
}

func TestOptionMissingFromTopNGetsTheLowestReturnedLogprob(t *testing.T) {
	answer := serve(t, newFake(map[string]float64{"A": -0.1, "B": -3}), nil).answer("q",
		map[string]any{"type": "choice", "instructions": "?", "criteria": map[string]any{"x": nil, "y": nil, "z": nil}})
	p := answer["probabilities"].(map[string]any)
	near(t, p["z"].(float64)/p["x"].(float64), math.Exp(-30+0.1))
}

func TestTooFewLogprobsIsABackendErrorNotAUniformAnswer(t *testing.T) {
	fake := newFake(map[string]float64{"A": -0.1})
	fake.short = true // a server that ignores `logprobs` returns only its top token
	status, body := serve(t, fake, nil).ask(map[string]any{"refund": noul})
	if status != 502 || !strings.Contains(body["error"].(map[string]any)["message"].(string), "logprobs") {
		t.Fatal(status, body)
	}
}

func TestLegacyLogprobShapeIsRead(t *testing.T) {
	fake := newFake(nil)
	fake.legacy = true
	near(t, serve(t, fake, nil).answer("refund", noul)["noul"].(float64), softmax2(-0.1, -2.5))
}

func TestCalibrationSoftensWithoutChangingTheChoice(t *testing.T) {
	question := map[string]any{"type": "choice", "instructions": "?", "criteria": map[string]any{"x": nil, "y": nil}}
	table := map[string]float64{"A": -0.05, "B": -3}
	raw := serve(t, newFake(table), nil).answer("q", question)
	calibrated := serve(t, newFake(table), func(c *ModelConfig) { c.CalibrationTemperature = 3 }).answer("q", question)
	x := calibrated["probabilities"].(map[string]any)["x"].(float64)
	if calibrated["choice"] != "x" || raw["choice"] != "x" || x <= 0.5 || x >= raw["probabilities"].(map[string]any)["x"].(float64) {
		t.Fatal(raw, calibrated)
	}
}

func TestDebiasAveragesBothOptionOrders(t *testing.T) {
	// The model always prefers the first position: a pure position bias.
	fake := newFake(nil)
	near(t, serve(t, fake, func(c *ModelConfig) { c.Debias = true }).answer("refund", noul)["noul"].(float64), 0.5)
	prompts := fake.prompt(0) + fake.prompt(1)
	if !strings.Contains(prompts, "A. No: Does not\nB. Yes: Asks for money back") {
		t.Fatal(prompts)
	}
}

func TestThinkingGeneratesReasoningThenReadsTheAnswerAfterIt(t *testing.T) {
	fake := newFake(nil)
	status, body := serve(t, fake, func(c *ModelConfig) { c.ThinkTokens = 256 }).ask(map[string]any{"refund": noul})
	think, read := fake.request[0], fake.request[1]
	if status != 200 || think["max_tokens"] != 256.0 || !strings.HasSuffix(think["prompt"].(string), "<think>\n") {
		t.Fatal(status, think)
	}
	if !strings.HasSuffix(read["prompt"].(string), "<think>\nIt depends on the receipt.\n</think>\n\n") {
		t.Fatal(read["prompt"])
	}
	if body["usage"].(map[string]any)["output_tokens"] != 8.0 {
		t.Fatal(body["usage"])
	}
}

func TestMoreThan26OptionsUseTwoLetterCodesScoredTokenByToken(t *testing.T) {
	fake := newFake(nil)
	fake.table = func(prompt string) map[string]float64 {
		switch {
		case strings.HasSuffix(prompt, "\n\nA"): // second letter after A
			return map[string]float64{"B": -0.2, "A": -2.0}
		case strings.HasSuffix(prompt, "\n\nB"):
			return map[string]float64{"A": -0.1}
		}
		return map[string]float64{"A": -0.3, "B": -1.5}
	}
	criteria := map[string]any{}
	for i := range 30 {
		criteria[string(rune('a'+i/26))+string(rune('a'+i%26))] = nil // aa..bd: sorted as AA..BD
	}
	answer := serve(t, fake, nil).answer("q", map[string]any{"type": "choice", "instructions": "?", "criteria": criteria})
	p := answer["probabilities"].(map[string]any)
	if strings.Contains(fake.prompt(0), "\nA. ") || !strings.Contains(fake.prompt(0), "\nAA. aa\nAB. ab") {
		t.Fatal(fake.prompt(0))
	}
	near(t, p["ab"].(float64)/p["aa"].(float64), math.Exp(-0.2+2.0))         // P(A)·P(B|A) vs P(A)·P(A|A)
	near(t, p["ba"].(float64)/p["ab"].(float64), math.Exp(-1.5-0.1+0.3+0.2)) // P(B)·P(A|B) vs P(A)·P(B|A)
	if len(fake.request) != 3 {                                              // the prompt, then the "A" and "B" prefixes
		t.Fatalf("%d completion requests", len(fake.request))
	}
}

func TestMoreThan26OptionsWithoutTokenizeIsAClear422(t *testing.T) {
	fake := newFake(nil)
	fake.tokens = false
	criteria := map[string]any{}
	for i := range 30 {
		criteria[strings.Repeat("o", i+1)] = nil
	}
	status, body := serve(t, fake, nil).ask(map[string]any{"q": map[string]any{"type": "choice", "instructions": "?", "criteria": criteria}})
	if status != 422 || !strings.Contains(body["error"].(map[string]any)["message"].(string), "/tokenize") {
		t.Fatal(status, body)
	}
}

func TestAliasesPathsAndModelsListing(t *testing.T) {
	c := serve(t, newFake(nil), nil)
	for _, path := range []string{"/api/v1/systemone", "/v1/systemone", "/api/alpha/decisions"} {
		status, body := c.do(http.MethodPost, path, map[string]any{"model": "jev-latest", "state": "s",
			"questions": map[string]any{"r": noul}})
		if status != 200 || body["model"] != "qwen-decision" {
			t.Fatal(path, status, body)
		}
	}
	_, models := c.do(http.MethodGet, "/v1/models", nil)
	data := models["data"].([]any)
	if len(data) != 2 || data[0].(map[string]any)["id"] != "qwen-decision" || data[1].(map[string]any)["alias_of"] != "qwen-decision" {
		t.Fatal(models)
	}
	if status, ready := c.do(http.MethodGet, "/ready", nil); status != 200 || ready["backends"].(map[string]any)["qwen-decision"] != true {
		t.Fatal(status, ready)
	}
	if status, _ := c.do(http.MethodGet, "/health", nil); status != 200 {
		t.Fatal(status)
	}
}

func TestErrorsMapToHTTPStatuses(t *testing.T) {
	c := serve(t, newFake(nil), nil)
	if status, _ := c.ask(map[string]any{"r": noul}, map[string]any{"model": "nope"}); status != 404 {
		t.Fatal(status)
	}
	if status, _ := c.ask(map[string]any{"r": map[string]any{"type": "score", "instructions": "?", "criteria": []any{"one"}}}); status != 422 {
		t.Fatal(status)
	}
	ignored := map[string]any{"provider": map[string]any{"order": []any{"x"}}, "session_id": "s", "user": "u"}
	if status, _ := c.ask(map[string]any{"r": noul}, ignored); status != 200 {
		t.Fatal(status)
	}
	if status, _ := serve(t, &fakeLlama{status: 500}, nil).ask(map[string]any{"r": noul}); status != 502 {
		t.Fatal(status)
	}
	down := serve(t, &fakeLlama{status: 500}, nil)
	if status, _ := down.do(http.MethodGet, "/ready", nil); status != 503 {
		t.Fatal(status)
	}
}

func TestBoxQuestionsAndImagesAreAClear422BeforeAnyBackendCall(t *testing.T) {
	image := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,iVBORw0KGgo="}}
	fake := newFake(nil)
	c := serve(t, fake, nil)
	status, body := c.ask(map[string]any{"cat": map[string]any{"type": "box", "instructions": "Where is the cat?"}},
		map[string]any{"model": "jev-latest"})
	if status != 422 || !strings.Contains(body["error"].(map[string]any)["message"].(string), "box") {
		t.Fatal(status, body)
	}
	status, body = c.ask(map[string]any{"r": noul}, map[string]any{"state": []any{map[string]any{"type": "text", "text": "Look"}, image}})
	if status != 422 || !strings.Contains(body["error"].(map[string]any)["message"].(string), "image") {
		t.Fatal(status, body)
	}
	inCriteria := map[string]any{"type": "choice", "instructions": "Which one?", "criteria": map[string]any{"a": []any{image}, "b": "text"}}
	if status, _ := c.ask(map[string]any{"r": inCriteria}); status != 422 {
		t.Fatal(status)
	}
	if len(fake.paths) != 0 {
		t.Fatal(fake.paths)
	}
}

func TestPromptIsByteIdenticalToTheBenchmarkedLabVersion(t *testing.T) {
	// Exactly what decision_bench.QwenServer sent during the JevBench runs in docs/benchmarks.md.
	expected := "<|im_start|>system\nChoose one option. Answer only with its code.\n<|im_end|>\n" +
		"<|im_start|>user\nState: {\"msg\":\"hi\"}\n\nQuestion: Greeting?\n\nOptions:\nA. Yes\nB. No" +
		"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
	state, _ := ParseJSON([]byte(`{"msg":"hi"}`))
	user := userMessage(state, Value{Kind: String, Text: "Greeting?"}, []string{"A", "B"}, []string{"Yes", "No"})
	empty := ""
	if got := chatML("Choose one option. Answer only with its code.", user, &empty); got != expected {
		t.Fatalf("%q", got)
	}
	if !strings.HasSuffix(chatML("s", "u", nil), "<think>\n") {
		t.Fatal("thinking must stay open")
	}
}
