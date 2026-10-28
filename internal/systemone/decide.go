package systemone

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
)

// Questions -> answer distributions -> System One answers.
//
// Each question is one prompt whose options are lettered A, B, C...; the answer distribution is
// the softmax of each code's logprob as the model's first answer token. Codes beyond Z need the
// server's tokenizer (llama-server /tokenize) and one extra request per shared token prefix.

// UnsupportedError: a valid request this backend cannot answer faithfully (HTTP 422).
type UnsupportedError struct{ msg string }

func (e *UnsupportedError) Error() string { return e.msg }

func unsupportedf(format string, args ...any) error {
	return &UnsupportedError{fmt.Sprintf(format, args...)}
}

func softmax(scores []float64) []float64 {
	top := math.Inf(-1)
	for _, s := range scores {
		top = math.Max(top, s)
	}
	weights := make([]float64, len(scores))
	total := 0.0
	for i, s := range scores {
		weights[i] = math.Exp(s - top)
		total += weights[i]
	}
	for i := range weights {
		weights[i] /= total
	}
	return weights
}

// calibrate: p ∝ p^(1/T). T > 1 softens; the argmax (hence accuracy) never changes.
func calibrate(probabilities []float64, temperature float64) []float64 {
	if temperature == 1 {
		return probabilities
	}
	scores := make([]float64, len(probabilities))
	for i, p := range probabilities {
		scores[i] = math.Log(math.Max(p, 1e-12)) / temperature
	}
	return softmax(scores)
}

// confidence is TypeSafe's: the top probability rescaled so a uniform distribution is 0.
// Equal to (n·p_max − 1)/(n − 1), i.e. p_max minus the mean of the others; it matches Jev's
// published answers and Contrastive-LM's reimplementation.
func confidence(probabilities []float64) float64 {
	n := float64(len(probabilities))
	top := 0.0
	for _, p := range probabilities {
		top = math.Max(top, p)
	}
	return math.Max(0, math.Min(1, (n*top-1)/(n-1)))
}

// topN indexes one completion's first-token alternatives.
type topN struct {
	byToken  map[string]float64
	byID     map[int]float64
	floor    float64 // lowest returned logprob: an upper bound for any option left out
	complete bool    // the server returned as many alternatives as requested
}

func newTopN(alternatives []alternative, requested int) topN {
	t := topN{byToken: map[string]float64{}, byID: map[int]float64{}, floor: math.Inf(1),
		complete: len(alternatives) >= requested}
	for _, alt := range alternatives {
		if _, seen := t.byToken[alt.Token]; !seen {
			t.byToken[alt.Token] = alt.Logprob
		}
		if alt.ID != nil {
			t.byID[*alt.ID] = alt.Logprob
		}
		t.floor = math.Min(t.floor, alt.Logprob)
	}
	return t
}

// lookup gives an option absent from the top-N the floor, but only when the top-N is complete:
// a server that ignores `logprobs` would otherwise make every option equal, silently.
func (t topN) lookup(logprob float64, found bool, code string) (float64, error) {
	if found {
		return logprob, nil
	}
	if !t.complete {
		return 0, backendErrorf("the backend returned fewer logprobs than requested and option %s is not among them; "+
			"it must honour `logprobs`", code)
	}
	return t.floor, nil
}

// codeLogprobs returns the log-probability of each code as the answer.
func codeLogprobs(ctx context.Context, backend *Backend, prompt string, letters []string, usage *Usage) ([]float64, error) {
	top := max(backend.Config.TopLogprobs, min(1000, 4*len(letters)))
	alternatives, err := backend.firstToken(ctx, prompt, usage, top)
	if err != nil {
		return nil, err
	}
	first := newTopN(alternatives, top)
	scores := make([]float64, len(letters))
	if len(letters[0]) == 1 {
		for i, code := range letters {
			logprob, found := first.byToken[code]
			if scores[i], err = first.lookup(logprob, found, code); err != nil {
				return nil, err
			}
		}
		return scores, nil
	}

	pieces := make([][]token, len(letters))
	for i, code := range letters {
		if pieces[i], err = backend.tokenize(ctx, code); err != nil {
			var failure *BackendError
			if errors.As(err, &failure) && (failure.Status == http.StatusNotFound || failure.Status == http.StatusMethodNotAllowed) {
				return nil, unsupportedf("more than 26 options needs llama-server /tokenize: %v", err)
			}
			return nil, err // a failing tokenizer is the backend's fault (502), not the request's
		}
		// Without pieces that spell the code, prefix requests would score the wrong text.
		var spelled strings.Builder
		for _, t := range pieces[i] {
			spelled.WriteString(t.Piece)
		}
		if spelled.String() != code {
			return nil, backendErrorf("/tokenize split %q into %d tokens spelling %q", code, len(pieces[i]), spelled.String())
		}
	}
	if len(first.byID) == 0 {
		return nil, unsupportedf("more than 26 options needs token ids in logprobs (direct llama-server access)")
	}
	cache := map[string]topN{"": first}
	distribution := func(prefix []token) (topN, error) {
		var key, text strings.Builder
		for _, t := range prefix {
			fmt.Fprintf(&key, "%d,", t.ID)
			text.WriteString(t.Piece)
		}
		if cached, ok := cache[key.String()]; ok {
			return cached, nil
		}
		alternatives, err := backend.firstToken(ctx, prompt+text.String(), usage, top)
		if err != nil {
			return topN{}, err
		}
		next := newTopN(alternatives, top)
		if len(next.byID) == 0 {
			return topN{}, unsupportedf("more than 26 options needs token ids in logprobs (direct llama-server access)")
		}
		cache[key.String()] = next
		return next, nil
	}
	for i, tokens := range pieces {
		for j, t := range tokens {
			dist, err := distribution(tokens[:j])
			if err != nil {
				return nil, err
			}
			logprob, found := dist.byID[t.ID]
			step, err := dist.lookup(logprob, found, letters[i])
			if err != nil {
				return nil, err
			}
			scores[i] += step
		}
	}
	return scores, nil
}

func optionDistribution(ctx context.Context, backend *Backend, state Value, q Question, labels []string,
	usage *Usage) ([]float64, error) {
	cfg := backend.Config
	letters := codes(len(labels))
	user := userMessage(state, q.Instructions, letters, labels)
	thinking := ""
	if cfg.ThinkTokens > 0 {
		var err error
		if thinking, err = backend.think(ctx, chatML(cfg.SystemPrompt, user, nil), usage); err != nil {
			return nil, err
		}
	}
	scores, err := codeLogprobs(ctx, backend, chatML(cfg.SystemPrompt, user, &thinking), letters, usage)
	if err != nil {
		return nil, err
	}
	return softmax(scores), nil
}

func answer(ctx context.Context, backend *Backend, state Value, q Question, usage *Usage) (object, error) {
	cfg := backend.Config
	keys, labels := options(q, cfg)
	probabilities, err := optionDistribution(ctx, backend, state, q, labels, usage)
	if err != nil {
		return nil, err
	}
	if q.Type == TypeNoul && cfg.Debias {
		// Small models favour a position regardless of content; averaging both orders cancels it.
		backward, err := optionDistribution(ctx, backend, state, q, []string{labels[1], labels[0]}, usage)
		if err != nil {
			return nil, err
		}
		probabilities = []float64{(probabilities[0] + backward[1]) / 2, (probabilities[1] + backward[0]) / 2}
	}
	probabilities = calibrate(probabilities, cfg.CalibrationTemperature)

	if q.Type == TypeNoul {
		return object{{"type", "noul"}, {"noul", probabilities[0]}}, nil
	}
	distribution := make(object, len(keys))
	best := 0
	for i, key := range keys {
		distribution[i] = field{key, probabilities[i]}
		if probabilities[i] > probabilities[best] {
			best = i
		}
	}
	if q.Type == TypeScore {
		// Probability-weighted level, as TypeSafe documents it: it can land between levels.
		score := 0.0
		legend := make(object, len(q.Levels))
		for level, p := range probabilities {
			score += float64(level) * p
			legend[level] = field{keys[level], render(q.Levels[level])}
		}
		return object{{"type", "score"}, {"score", score}, {"legend", legend},
			{"probabilities", distribution}, {"confidence", confidence(probabilities)}}, nil
	}
	return object{{"type", "choice"}, {"choice", keys[best]}, {"probabilities", distribution},
		{"confidence", confidence(probabilities)}}, nil
}

// checkSupported refuses what a text-only model would otherwise answer from garbage (an
// image's URL or base64 pasted into the prompt).
func checkSupported(request Request) error {
	for _, q := range request.Questions {
		if q.Type == TypeBox {
			return unsupportedf("question %q: box questions need image input, which this text-only model lacks", q.Key)
		}
	}
	images := hasImage(request.State)
	for _, q := range request.Questions {
		images = images || hasImage(q.Instructions) || hasImage(Value{Kind: Object, Members: q.Options}) ||
			hasImage(Value{Kind: Array, Items: q.Levels})
		for _, criterion := range []*Value{q.True, q.False} {
			images = images || (criterion != nil && hasImage(*criterion))
		}
	}
	if images {
		return unsupportedf("image input is not supported by this text-only model")
	}
	return nil
}

// Decide answers every question of a request concurrently, within the backend's concurrency.
func Decide(ctx context.Context, settings *Settings, backends map[string]*Backend, model NamedModel,
	request Request) (object, error) {
	if err := checkSupported(request); err != nil {
		return nil, err
	}
	backend := backends[model.Name]
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var usage Usage
	answers := make(object, len(request.Questions))
	errs := make([]error, len(request.Questions))
	// Only as many questions in flight as backend slots: each prompt repeats the whole state,
	// so 255 questions built at once would hold 255 copies of it.
	workers := make(chan struct{}, backend.Config.Concurrency)
	var wg sync.WaitGroup
	for i, q := range request.Questions {
		wg.Go(func() {
			select {
			case workers <- struct{}{}:
				defer func() { <-workers }()
			case <-ctx.Done():
				errs[i] = ctx.Err()
				return
			}
			a, err := answer(ctx, backend, request.State, q.Question, &usage)
			if err != nil {
				errs[i] = err
				cancel()
			}
			answers[i] = field{q.Key, a}
		})
	}
	wg.Wait()
	if err := firstCause(errs); err != nil {
		return nil, err
	}
	id := make([]byte, 16)
	_, _ = rand.Read(id)
	return object{
		{"id", "gen-dec-" + hex.EncodeToString(id)},
		{"model", model.Name},
		{"provider", settings.Provider},
		{"answers", answers},
		{"usage", object{{"input_tokens", usage.InputTokens.Load()}, {"output_tokens", usage.OutputTokens.Load()},
			{"cost", 0}}},
	}, nil
}

// firstCause prefers the error that failed the request over the cancellations it caused.
func firstCause(errs []error) error {
	var fallback error
	for _, err := range errs {
		if err == nil {
			continue
		}
		if !errors.Is(err, context.Canceled) {
			return err
		}
		fallback = err
	}
	return fallback
}
