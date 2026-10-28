package systemone

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// Kind is the type of a JSON value.
type Kind int

const (
	Null Kind = iota
	Bool
	Number
	String
	Array
	Object
)

// Member is one key of a JSON object.
type Member struct {
	Key   string
	Value Value
}

// Value is a decoded JSON value that keeps what Python's json module keeps, so prompts stay
// byte-identical to the benchmarked Python version: key order (a repeated key keeps its first
// position and its last value) and number literals.
type Value struct {
	Kind    Kind
	Bool    bool
	Text    string // String: the text; Number: the literal
	Items   []Value
	Members []Member
}

// maxDepth bounds nesting, as Python's recursion limit does.
const maxDepth = 512

// ParseJSON decodes exactly one JSON value.
func ParseJSON(data []byte) (Value, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	value, err := decodeValue(dec, 0)
	if err != nil {
		return Value{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return Value{}, errors.New("trailing data after the JSON value")
	}
	return value, nil
}

func decodeValue(dec *json.Decoder, depth int) (Value, error) {
	if depth > maxDepth {
		return Value{}, errors.New("JSON nested too deeply")
	}
	token, err := dec.Token()
	if err != nil {
		return Value{}, err
	}
	switch t := token.(type) {
	case nil:
		return Value{Kind: Null}, nil
	case bool:
		return Value{Kind: Bool, Bool: t}, nil
	case json.Number:
		return Value{Kind: Number, Text: string(t)}, nil
	case string:
		return Value{Kind: String, Text: t}, nil
	case json.Delim:
		value := Value{Kind: Array}
		if t == '{' {
			value.Kind = Object
		}
		index := map[string]int{}
		for dec.More() {
			if value.Kind == Array {
				item, err := decodeValue(dec, depth+1)
				if err != nil {
					return Value{}, err
				}
				value.Items = append(value.Items, item)
				continue
			}
			key, err := dec.Token()
			if err != nil {
				return Value{}, err
			}
			item, err := decodeValue(dec, depth+1)
			if err != nil {
				return Value{}, err
			}
			if i, seen := index[key.(string)]; seen {
				value.Members[i].Value = item
			} else {
				index[key.(string)] = len(value.Members)
				value.Members = append(value.Members, Member{key.(string), item})
			}
		}
		if _, err := dec.Token(); err != nil { // closing delimiter
			return Value{}, err
		}
		return value, nil
	}
	return Value{}, fmt.Errorf("unexpected JSON token %v", token)
}

// Get returns an object's member.
func (v Value) Get(key string) (Value, bool) {
	for _, member := range v.Members {
		if member.Key == key {
			return member.Value, true
		}
	}
	return Value{}, false
}

// PythonJSON renders v as json.dumps(v, ensure_ascii=False, separators=(",", ":")) does.
func (v Value) PythonJSON() string {
	var b strings.Builder
	v.writePython(&b)
	return b.String()
}

func (v Value) writePython(b *strings.Builder) {
	switch v.Kind {
	case Null:
		b.WriteString("null")
	case Bool:
		b.WriteString(strconv.FormatBool(v.Bool))
	case Number:
		b.WriteString(pythonNumber(v.Text))
	case String:
		writePythonString(b, v.Text)
	case Array:
		b.WriteByte('[')
		for i, item := range v.Items {
			if i > 0 {
				b.WriteByte(',')
			}
			item.writePython(b)
		}
		b.WriteByte(']')
	case Object:
		b.WriteByte('{')
		for i, member := range v.Members {
			if i > 0 {
				b.WriteByte(',')
			}
			writePythonString(b, member.Key)
			b.WriteByte(':')
			member.Value.writePython(b)
		}
		b.WriteByte('}')
	}
}

// pythonNumber: Python parses a literal with a fraction or exponent as a float and the rest as
// an exact integer.
func pythonNumber(literal string) string {
	if strings.ContainsAny(literal, ".eE") {
		f, _ := strconv.ParseFloat(literal, 64) // out of range gives ±Inf, as in Python
		return pythonFloat(f)
	}
	if literal == "-0" {
		return "0"
	}
	return literal
}

// pythonFloat is Python's float repr: shortest round-trip digits, scientific notation below
// 1e-4 and from 1e16, and always a fraction otherwise.
func pythonFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case f == 0 && math.Signbit(f):
		return "-0.0"
	case f == 0:
		return "0.0"
	}
	s := strconv.FormatFloat(f, 'e', -1, 64)
	sign := ""
	if s[0] == '-' {
		sign, s = "-", s[1:]
	}
	mantissa, exponent, _ := strings.Cut(s, "e")
	exp, _ := strconv.Atoi(exponent)
	digits := strings.Replace(mantissa, ".", "", 1)
	switch {
	case exp < -4 || exp >= 16:
		return fmt.Sprintf("%s%se%+03d", sign, mantissa, exp)
	case exp < 0:
		return sign + "0." + strings.Repeat("0", -exp-1) + digits
	case len(digits) <= exp+1:
		return sign + digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
	default:
		return sign + digits[:exp+1] + "." + digits[exp+1:]
	}
}

// writePythonString escapes like Python with ensure_ascii=False: quotes, backslashes and
// control characters only.
func writePythonString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// hasImage reports an image part ({"type": "image_url", ...}) at any depth.
func hasImage(v Value) bool {
	switch v.Kind {
	case Object:
		if kind, ok := v.Get("type"); ok && kind.Kind == String && kind.Text == "image_url" {
			return true
		}
		for _, member := range v.Members {
			if hasImage(member.Value) {
				return true
			}
		}
	case Array:
		for _, item := range v.Items {
			if hasImage(item) {
				return true
			}
		}
	}
	return false
}
