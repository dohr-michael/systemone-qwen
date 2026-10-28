package systemone

import (
	"errors"
	"fmt"
	"os"

	"go.yaml.in/yaml/v3"
)

// Model registry: which llama-server answers which `model` name, and how to ask it.
// Configuration comes from a YAML file (SYSTEMONE_CONFIG) or, for a single model, from
// environment variables. See config.example.yaml for every field.

type ModelConfig struct {
	// llama-server root URL (direct access enables /tokenize, hence more than 26 options).
	BaseURL string `yaml:"base_url"`
	// `model` sent to the backend: llama-server ignores it, LiteLLM routes on it.
	ServedModel string `yaml:"served_model"`
	// Name of the environment variable holding the backend API key, if any.
	APIKeyEnv string `yaml:"api_key_env"`
	// How noul options are worded: "Yes"/"No", the criteria, or "Yes: <criterion>".
	NoulLabels string `yaml:"noul_labels"`
	// Average noul probabilities over both option orders; only helps sub-1B models.
	Debias bool `yaml:"debias"`
	// p ∝ p^(1/T) on each answer distribution; >1 softens over-confident models.
	CalibrationTemperature float64 `yaml:"calibration_temperature"`
	// >0: let the model reason up to that many tokens before reading the answer.
	ThinkTokens int `yaml:"think_tokens"`
	// First-token alternatives requested; options missing from them get the lowest returned logprob.
	TopLogprobs int `yaml:"top_logprobs"`
	// Parallel backend requests per decision request (match llama-server --parallel).
	Concurrency int     `yaml:"concurrency"`
	TimeoutS    float64 `yaml:"timeout_s"`
	// Extra fields merged into every completion request (llama-server: reuse the KV prefix).
	ExtraBody    map[string]any `yaml:"extra_body"`
	SystemPrompt string         `yaml:"system_prompt"`
}

func defaultModelConfig() ModelConfig {
	return ModelConfig{
		NoulLabels:             "prefixed",
		CalibrationTemperature: 1,
		TopLogprobs:            100,
		Concurrency:            1,
		TimeoutS:               120,
		SystemPrompt:           "Choose one option. Answer only with its code.",
	}
}

func (c *ModelConfig) validate() error {
	switch {
	case c.BaseURL == "":
		return errors.New("base_url is required")
	case c.NoulLabels != "plain" && c.NoulLabels != "criteria" && c.NoulLabels != "prefixed":
		return fmt.Errorf("noul_labels must be plain, criteria or prefixed, not %q", c.NoulLabels)
	case c.CalibrationTemperature <= 0:
		return errors.New("calibration_temperature must be > 0")
	case c.ThinkTokens < 0:
		return errors.New("think_tokens must be >= 0")
	case c.TopLogprobs < 2:
		return errors.New("top_logprobs must be >= 2")
	case c.Concurrency < 1:
		return errors.New("concurrency must be >= 1")
	case c.TimeoutS <= 0:
		return errors.New("timeout_s must be > 0")
	}
	if c.ExtraBody == nil {
		c.ExtraBody = map[string]any{"cache_prompt": true}
	}
	return nil
}

type NamedModel struct {
	Name   string
	Config ModelConfig
}

type Alias struct {
	Name, Target string
}

type Settings struct {
	// In file order: the first model is the default unless DefaultModel names another.
	Models []NamedModel
	// Extra names accepted in `model`, e.g. jev-latest -> qwen-decision.
	Aliases []Alias
	// Used when a request names no model, or one that is not configured, if AllowUnknownModel.
	DefaultModel      string
	AllowUnknownModel bool
	// Name reported in the response's `provider` field.
	Provider string
}

// ParseSettings reads a YAML registry (JSON works too: it is YAML).
func ParseSettings(data []byte) (*Settings, error) {
	var file struct {
		Models            yaml.Node `yaml:"models"`
		Aliases           yaml.Node `yaml:"aliases"`
		DefaultModel      string    `yaml:"default_model"`
		AllowUnknownModel bool      `yaml:"allow_unknown_model"`
		Provider          string    `yaml:"provider"`
	}
	if err := yaml.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	settings := &Settings{DefaultModel: file.DefaultModel, AllowUnknownModel: file.AllowUnknownModel,
		Provider: file.Provider}
	err := eachPair(&file.Models, func(name string, node *yaml.Node) error {
		cfg := defaultModelConfig()
		if err := node.Decode(&cfg); err != nil {
			return fmt.Errorf("models.%s: %w", name, err)
		}
		settings.Models = append(settings.Models, NamedModel{name, cfg})
		return nil
	})
	if err != nil {
		return nil, err
	}
	err = eachPair(&file.Aliases, func(name string, node *yaml.Node) error {
		settings.Aliases = append(settings.Aliases, Alias{name, node.Value})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return settings, settings.validate()
}

// eachPair walks a YAML mapping in file order; an absent node is an empty mapping.
func eachPair(node *yaml.Node, visit func(string, *yaml.Node) error) error {
	if node.Kind == 0 || node.Tag == "!!null" {
		return nil
	}
	if node.Kind != yaml.MappingNode {
		return fmt.Errorf("line %d: expected a mapping", node.Line)
	}
	for i := 0; i < len(node.Content); i += 2 {
		if err := visit(node.Content[i].Value, node.Content[i+1]); err != nil {
			return err
		}
	}
	return nil
}

func (s *Settings) validate() error {
	if len(s.Models) == 0 {
		return errors.New("at least one model must be configured")
	}
	for i := range s.Models {
		if err := s.Models[i].Config.validate(); err != nil {
			return fmt.Errorf("models.%s: %w", s.Models[i].Name, err)
		}
	}
	for _, alias := range s.Aliases {
		if s.model(alias.Target) < 0 {
			return fmt.Errorf("alias %q points to unknown model %q", alias.Name, alias.Target)
		}
	}
	if s.DefaultModel == "" {
		s.DefaultModel = s.Models[0].Name
	} else if s.model(s.DefaultModel) < 0 {
		return fmt.Errorf("default_model %q is not configured", s.DefaultModel)
	}
	if s.Provider == "" {
		s.Provider = "systemone-qwen"
	}
	return nil
}

func (s *Settings) model(name string) int {
	for i, m := range s.Models {
		if m.Name == name {
			return i
		}
	}
	return -1
}

// Resolve maps a requested name (or alias) to a configured model; false when the name is
// unknown and not allowed.
func (s *Settings) Resolve(name string) (NamedModel, bool) {
	if name != "" {
		for _, alias := range s.Aliases {
			if alias.Name == name {
				name = alias.Target
				break
			}
		}
		if i := s.model(name); i >= 0 {
			return s.Models[i], true
		}
		if !s.AllowUnknownModel {
			return NamedModel{}, false
		}
	}
	return s.Models[s.model(s.DefaultModel)], true
}

// LoadSettings reads SYSTEMONE_CONFIG or, for a single llama-server, SYSTEMONE_BACKEND_URL.
func LoadSettings(getenv func(string) string) (*Settings, error) {
	if path := getenv("SYSTEMONE_CONFIG"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		return ParseSettings(data)
	}
	baseURL := getenv("SYSTEMONE_BACKEND_URL")
	if baseURL == "" {
		return nil, errors.New("set SYSTEMONE_CONFIG (YAML file) or SYSTEMONE_BACKEND_URL (single llama-server)")
	}
	name := getenv("SYSTEMONE_MODEL")
	if name == "" {
		name = "qwen-decision"
	}
	cfg := defaultModelConfig()
	cfg.BaseURL = baseURL
	cfg.ServedModel = getenv("SYSTEMONE_SERVED_MODEL")
	cfg.APIKeyEnv = getenv("SYSTEMONE_API_KEY_ENV")
	settings := &Settings{Models: []NamedModel{{name, cfg}}}
	return settings, settings.validate()
}
