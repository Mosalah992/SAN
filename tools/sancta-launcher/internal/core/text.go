package core

import (
	"regexp"
	"strings"
)

var ansiSeqRe = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]`)

// SanitizeLogText removes ANSI escape sequences and strips disruptive control
// bytes so launcher logs stay readable in the GUI and CLI.
func SanitizeLogText(s string) string {
	if s == "" {
		return ""
	}
	s = ansiSeqRe.ReplaceAllString(s, "")
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r == '\n' || r == '\t' || r == '\r':
			b.WriteRune(r)
		case r >= 32:
			b.WriteRune(r)
		}
	}
	out := strings.TrimSpace(b.String())
	out = strings.ReplaceAll(out, "\r", " ")
	out = strings.ReplaceAll(out, "\n", " ")
	return strings.Join(strings.Fields(out), " ")
}
