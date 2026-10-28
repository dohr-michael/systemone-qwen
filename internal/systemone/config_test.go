package systemone

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPythonFloatRepr(t *testing.T) {
	for f, want := range map[float64]string{
		100: "100.0", 1.5: "1.5", 1e-5: "1e-05", 1e-4: "0.0001", 1e16: "1e+16", 1e15: "1000000000000000.0",
		1.2345678901234568e+17: "1.2345678901234568e+17", 5e-324: "5e-324", 1e300: "1e+300", -2.5e-7: "-2.5e-07",
		math.Copysign(0, -1): "-0.0", 0: "0.0", math.Inf(1): "Infinity", 0.1: "0.1",
	} {
		if got := pythonFloat(f); got != want {
			t.Errorf("pythonFloat(%v) = %s, want %s", f, got, want)
		}
	}
}

func TestParseJSONKeepsPythonSemantics(t *testing.T) {
	v, err := ParseJSON([]byte(`{"b":1,"a":2,"b":3,"n":-0,"big":12345678901234567890,"s":"<&>\u0007"}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := v.PythonJSON(); got != `{"b":3,"a":2,"n":0,"big":12345678901234567890,"s":"<&>\u0007"}` {
		t.Fatal(got)
	}
	for _, bad := range []string{`{"a":1} x`, `{"a":`, `NaN`, strings.Repeat("[", maxDepth+2) + strings.Repeat("]", maxDepth+2)} {
		if _, err := ParseJSON([]byte(bad)); err == nil {
			t.Errorf("%.20q: want an error", bad)
		}
	}
}

func TestSettingsKeepFileOrderAndDefaults(t *testing.T) {
	settings, err := ParseSettings([]byte(`
aliases: {jev-latest: zeta}
models:
  zeta: {base_url: "http://z:8080", think_tokens: 512}
  alpha: {base_url: "http://a:8080", extra_body: {n_keep: 0}}
`))
	if err != nil {
		t.Fatal(err)
	}
	if settings.DefaultModel != "zeta" || settings.Models[1].Name != "alpha" || settings.Provider != "systemone-qwen" {
		t.Fatal(settings)
	}
	zeta, alpha := settings.Models[0].Config, settings.Models[1].Config
	if zeta.ThinkTokens != 512 || zeta.TopLogprobs != 100 || zeta.ExtraBody["cache_prompt"] != true {
		t.Fatal(zeta)
	}
	if _, merged := alpha.ExtraBody["cache_prompt"]; merged || alpha.ExtraBody["n_keep"] != 0 {
		t.Fatal("extra_body replaces the default, as before:", alpha.ExtraBody)
	}
	if m, ok := settings.Resolve("jev-latest"); !ok || m.Name != "zeta" {
		t.Fatal(m)
	}
	if _, ok := settings.Resolve("nope"); ok {
		t.Fatal("unknown models are refused by default")
	}
}

func TestSettingsErrors(t *testing.T) {
	for _, bad := range []string{
		`models: {}`,
		`models: {m: {}}`,
		`models: {m: {base_url: x, noul_labels: loud}}`,
		`models: {m: {base_url: x, calibration_temperature: 0}}`,
		`{models: {m: {base_url: x}}, aliases: {a: missing}}`,
		`{models: {m: {base_url: x}}, default_model: missing}`,
	} {
		if _, err := ParseSettings([]byte(bad)); err == nil {
			t.Errorf("%s: want an error", bad)
		}
	}
}

func TestLoadSettingsFromTheEnvironmentOrAFile(t *testing.T) {
	env := map[string]string{"SYSTEMONE_BACKEND_URL": "http://llama:8080", "SYSTEMONE_SERVED_MODEL": "qwen"}
	settings, err := LoadSettings(func(k string) string { return env[k] })
	if err != nil || settings.Models[0].Name != "qwen-decision" || settings.Models[0].Config.ServedModel != "qwen" {
		t.Fatal(settings, err)
	}
	if _, err := LoadSettings(func(string) string { return "" }); err == nil {
		t.Fatal("no configuration must fail")
	}
	example, err := filepath.Abs("../../config.example.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(example); err != nil {
		t.Skip("no config.example.yaml")
	}
	settings, err = LoadSettings(func(k string) string { return map[string]string{"SYSTEMONE_CONFIG": example}[k] })
	if err != nil || len(settings.Models) != 2 || settings.Models[0].Config.CalibrationTemperature != 1.5 {
		t.Fatal(settings, err)
	}
}
