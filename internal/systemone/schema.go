package systemone

import (
	"fmt"
)

// Question types of the System One wire format (OpenRouter /api/v1/systemone, TypeSafe /v1/systemone).
const (
	TypeNoul   = "noul"
	TypeChoice = "choice"
	TypeScore  = "score"
	TypeBox    = "box" // Jev 1.14: locate one target in an image. Parsed only to be refused.
)

// Question is one validated question. Instructions and criteria accept a string, an object or
// an array (TypeSafe "Advanced: structure").
type Question struct {
	Type         string
	Instructions Value
	// noul: the optional true/false criteria (nil when absent).
	True, False *Value
	// choice: option key -> description; Null means the key speaks for itself.
	Options []Member
	// score: ordered levels.
	Levels []Value
}

// Request is a validated decision request. OpenRouter-only fields (provider, session_id,
// user, trace) are accepted and ignored.
type Request struct {
	Model     string
	State     Value
	Questions []NamedQuestion
}

type NamedQuestion struct {
	Key string
	Question
}

// ParseRequest decodes and validates a request body with the limits TypeSafe documents.
func ParseRequest(body []byte) (Request, error) {
	root, err := ParseJSON(body)
	if err != nil {
		return Request{}, fmt.Errorf("invalid JSON: %w", err)
	}
	if root.Kind != Object {
		return Request{}, fmt.Errorf("the request must be a JSON object")
	}
	var request Request
	if model, ok := root.Get("model"); ok && model.Kind != Null {
		if model.Kind != String {
			return Request{}, fmt.Errorf("model: must be a string")
		}
		request.Model = model.Text
	}
	state, ok := root.Get("state")
	if !ok {
		return Request{}, fmt.Errorf("state: field required")
	}
	request.State = state
	questions, ok := root.Get("questions")
	if !ok || questions.Kind != Object {
		return Request{}, fmt.Errorf("questions: an object is required")
	}
	if n := len(questions.Members); n < 1 || n > 255 {
		return Request{}, fmt.Errorf("questions: 1 to 255 questions, got %d", n)
	}
	for _, member := range questions.Members {
		question, err := parseQuestion(member.Value)
		if err != nil {
			return Request{}, fmt.Errorf("questions.%s.%w", member.Key, err)
		}
		request.Questions = append(request.Questions, NamedQuestion{member.Key, question})
	}
	return request, nil
}

func parseQuestion(raw Value) (Question, error) {
	if raw.Kind != Object {
		return Question{}, fmt.Errorf("type: the question must be an object")
	}
	kind, _ := raw.Get("type")
	question := Question{Type: kind.Text}
	switch {
	case kind.Kind != String:
		return Question{}, fmt.Errorf("type: required, one of noul, choice, score, box")
	case question.Type != TypeNoul && question.Type != TypeChoice && question.Type != TypeScore &&
		question.Type != TypeBox:
		return Question{}, fmt.Errorf("type: unknown question type %q", question.Type)
	}
	instructions, ok := raw.Get("instructions")
	if !ok || !isContent(instructions) {
		return Question{}, fmt.Errorf("instructions: a string, object or array is required")
	}
	question.Instructions = instructions
	criteria, _ := raw.Get("criteria")

	switch question.Type {
	case TypeNoul:
		if criteria.Kind == Null {
			return question, nil
		}
		if criteria.Kind != Object {
			return Question{}, fmt.Errorf("criteria: must be an object with optional true/false")
		}
		for name, target := range map[string]**Value{"true": &question.True, "false": &question.False} {
			value, ok := criteria.Get(name)
			if !ok || value.Kind == Null {
				continue
			}
			if !isContent(value) {
				return Question{}, fmt.Errorf("criteria.%s: must be a string, object or array", name)
			}
			*target = &value
		}
	case TypeChoice:
		if criteria.Kind != Object || len(criteria.Members) < 2 || len(criteria.Members) > 255 {
			return Question{}, fmt.Errorf("criteria: an object of 2 to 255 options is required")
		}
		for _, option := range criteria.Members {
			if option.Value.Kind != Null && !isContent(option.Value) {
				return Question{}, fmt.Errorf("criteria.%s: must be null, a string, an object or an array", option.Key)
			}
		}
		question.Options = criteria.Members
	case TypeScore:
		if criteria.Kind != Array || len(criteria.Items) < 2 || len(criteria.Items) > 10 {
			return Question{}, fmt.Errorf("criteria: a list of 2 to 10 levels is required")
		}
		for i, level := range criteria.Items {
			if !isContent(level) {
				return Question{}, fmt.Errorf("criteria.%d: must be a string, object or array", i)
			}
		}
		question.Levels = criteria.Items
	}
	return question, nil
}

func isContent(v Value) bool {
	return v.Kind == String || v.Kind == Object || v.Kind == Array
}
