package systemone

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// HTTP API: System One decisions over the OpenRouter / TypeSafe wire format.

// field and object keep JSON keys in order, as clients see them from Jev.
type field struct {
	Key   string
	Value any
}

type object []field

func (o object) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		key, err := json.Marshal(f.Key)
		if err != nil {
			return nil, err
		}
		value, err := json.Marshal(f.Value)
		if err != nil {
			return nil, err
		}
		b.Write(key)
		b.WriteByte(':')
		b.Write(value)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// maxBody is far above Jev's 262k-token context.
const maxBody = 16 << 20

type server struct {
	settings *Settings
	backends map[string]*Backend
}

// NewHandler serves the API; backends maps each configured model name to its llama-server.
func NewHandler(settings *Settings, backends map[string]*Backend, logger *slog.Logger) http.Handler {
	s := &server{settings, backends}
	mux := http.NewServeMux()
	// OpenRouter SDKs call {server_url}/systemone with server_url ending in /api/v1; TypeSafe's
	// own path is /v1/systemone; /api/alpha/decisions is OpenRouter's earlier alpha path.
	for _, path := range []string{"/api/v1/systemone", "/v1/systemone", "/api/alpha/decisions"} {
		mux.HandleFunc("POST "+path, s.decide)
	}
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /api/v1/models", s.models)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, object{{"status", "ok"}})
	})
	mux.HandleFunc("GET /ready", s.ready)
	return logRequests(logger, mux)
}

func (s *server) decide(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("request body over %d bytes", maxBody))
		return
	}
	request, err := ParseRequest(body)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	model, ok := s.settings.Resolve(request.Model)
	if !ok {
		writeError(w, http.StatusNotFound, fmt.Sprintf("unknown model %q; see GET /v1/models", request.Model))
		return
	}
	decision, err := Decide(r.Context(), s.settings, s.backends, model, request)
	var unsupported *UnsupportedError
	var failure *BackendError
	switch {
	case err == nil:
		writeJSON(w, http.StatusOK, decision)
	case errors.As(err, &unsupported):
		writeError(w, http.StatusUnprocessableEntity, err.Error())
	case errors.As(err, &failure):
		writeError(w, http.StatusBadGateway, "model backend failed: "+err.Error())
	case r.Context().Err() != nil:
		// The client left; nobody reads the answer.
	default:
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}

func (s *server) models(w http.ResponseWriter, _ *http.Request) {
	data := []object{}
	for _, m := range s.settings.Models {
		data = append(data, object{{"id", m.Name}, {"object", "model"}, {"owned_by", s.settings.Provider}})
	}
	for _, a := range s.settings.Aliases {
		data = append(data, object{{"id", a.Name}, {"object", "model"}, {"owned_by", s.settings.Provider},
			{"alias_of", a.Target}})
	}
	writeJSON(w, http.StatusOK, object{{"object", "list"}, {"data", data}})
}

func (s *server) ready(w http.ResponseWriter, r *http.Request) {
	status, code := object{}, http.StatusOK
	for _, m := range s.settings.Models {
		healthy := s.backends[m.Name].Healthy(r.Context())
		if !healthy {
			code = http.StatusServiceUnavailable
		}
		status = append(status, field{m.Name, healthy})
	}
	writeJSON(w, code, object{{"backends", status}})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	data, err := json.Marshal(body)
	if err != nil {
		status, data = http.StatusInternalServerError, []byte(`{"error":{"message":"unencodable answer","code":500}}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(data)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, object{{"error", object{{"message", message}, {"code", status}}}})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func logRequests(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		recorder := &statusRecorder{w, http.StatusOK}
		next.ServeHTTP(recorder, r)
		logger.Info("request", "method", r.Method, "path", r.URL.Path, "status", recorder.status,
			"duration_ms", time.Since(start).Milliseconds())
	})
}
