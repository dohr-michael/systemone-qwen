package systemone

import (
	"strconv"
	"strings"
	"unicode"
)

// Prompt construction, kept byte-identical to the benchmarked version (see docs/benchmarks.md).
// The model sees the state, the question and lettered options, and answers with one code; the
// answer is read from the first token's logprobs, never generated. Thinking is closed unless a
// reasoning budget is configured.

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

// render: strings verbatim, structure as compact JSON.
func render(v Value) string {
	if v.Kind == String {
		return v.Text
	}
	return v.PythonJSON()
}

// codes letters options A..Z, or AA, AB... for every option beyond 26: codes of one length
// keep a code from being the prefix of another, which would inflate the shorter one.
func codes(count int) []string {
	out := make([]string, 0, count)
	for i := range count {
		if count <= len(alphabet) {
			out = append(out, alphabet[i:i+1])
		} else {
			out = append(out, alphabet[i/26:i/26+1]+alphabet[i%26:i%26+1])
		}
	}
	return out
}

// options -> (answer keys in order, option labels shown to the model).
func options(q Question, cfg ModelConfig) ([]string, []string) {
	switch q.Type {
	case TypeNoul:
		keys := []string{"true", "false"}
		switch {
		case cfg.NoulLabels == "plain" || q.True == nil || q.False == nil:
			return keys, []string{"Yes", "No"}
		case cfg.NoulLabels == "prefixed":
			return keys, []string{"Yes: " + render(*q.True), "No: " + render(*q.False)}
		default:
			return keys, []string{render(*q.True), render(*q.False)}
		}
	case TypeChoice:
		keys := make([]string, len(q.Options))
		labels := make([]string, len(q.Options))
		for i, option := range q.Options {
			keys[i] = option.Key
			// Key and description, as Jev sees them: keys often carry meaning ("billing").
			labels[i] = option.Key
			if option.Value.Kind != Null && (option.Value.Kind != String || option.Value.Text != "") {
				labels[i] = option.Key + ": " + render(option.Value)
			}
		}
		return keys, labels
	}
	keys := make([]string, len(q.Levels))
	labels := make([]string, len(q.Levels))
	for i, level := range q.Levels {
		keys[i] = strconv.Itoa(i)
		labels[i] = render(level)
	}
	return keys, labels
}

func userMessage(state, instructions Value, letters, labels []string) string {
	lines := make([]string, len(letters))
	for i := range letters {
		lines[i] = letters[i] + ". " + labels[i]
	}
	return "State: " + state.PythonJSON() + "\n\nQuestion: " + render(instructions) +
		"\n\nOptions:\n" + strings.Join(lines, "\n")
}

// chatML builds Qwen's ChatML. A nil thinking leaves the think block open so the model can
// generate its reasoning first; otherwise the block is closed after that (possibly empty) reasoning.
func chatML(system, user string, thinking *string) string {
	prompt := "<|im_start|>system\n" + system + "\n<|im_end|>\n" +
		"<|im_start|>user\n" + user + "<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n"
	if thinking == nil {
		return prompt
	}
	if reasoning := pythonStrip(*thinking); reasoning != "" {
		return prompt + reasoning + "\n</think>\n\n"
	}
	return prompt + "\n</think>\n\n"
}

// pythonStrip trims what Python's str.strip() does: Unicode whitespace plus \x1c-\x1f.
func pythonStrip(s string) string {
	return strings.TrimFunc(s, func(r rune) bool { return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f) })
}
