// Command systemone-qwen serves System One decisions from a Qwen model on llama-server.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/dohr-michael/systemone-qwen/internal/systemone"
)

func main() {
	if err := run(); err != nil {
		slog.Error("systemone-qwen stopped", "error", err)
		os.Exit(1)
	}
}

func run() error {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel(os.Getenv("SYSTEMONE_LOG_LEVEL"))}))
	settings, err := systemone.LoadSettings(os.Getenv)
	if err != nil {
		return err
	}
	backends := map[string]*systemone.Backend{}
	for _, m := range settings.Models {
		backends[m.Name] = systemone.NewBackend(m.Config, nil)
	}
	address := net.JoinHostPort(env("SYSTEMONE_HOST", "0.0.0.0"), env("SYSTEMONE_PORT", "8000"))
	server := &http.Server{
		Addr:              address,
		Handler:           systemone.NewHandler(settings, backends, logger),
		ReadHeaderTimeout: 10 * time.Second,
		// The whole request, body included; no write timeout: thinking can take minutes.
		ReadTimeout: 60 * time.Second,
		IdleTimeout: 120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	failed := make(chan error, 1)
	go func() { failed <- server.ListenAndServe() }()
	logger.Info("listening", "address", address, "models", len(settings.Models))
	select {
	case err := <-failed:
		return err
	case <-ctx.Done():
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdown); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func logLevel(name string) slog.Level {
	switch strings.ToLower(name) {
	case "debug", "trace":
		return slog.LevelDebug
	case "warning", "warn":
		return slog.LevelWarn
	case "error", "critical":
		return slog.LevelError
	}
	return slog.LevelInfo
}
