package core

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// RunLogTail polls log files (same set as Python launcher) and emits lines.
func (m *Manager) RunLogTail(ctx context.Context) {
	root := m.Paths.Root
	files := map[string]string{
		"security":   filepath.Join(root, "logs", "security.jsonl"),
		"redteam":    filepath.Join(root, "logs", "red_team.jsonl"),
		"behavioral": filepath.Join(root, "logs", "behavioral.jsonl"),
		"activity":   filepath.Join(root, "logs", "agent_activity.log"),
	}
	cursors := make(map[string]int64)
	for k, p := range files {
		if st, err := os.Stat(p); err == nil {
			cursors[k] = st.Size()
		}
	}
	t := time.NewTicker(2 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			for src, p := range files {
				m.tailFile(src, p, cursors)
			}
		}
	}
}

func (m *Manager) tailFile(src, path string, cursors map[string]int64) {
	st, err := os.Stat(path)
	if err != nil {
		return
	}
	size := st.Size()
	off := cursors[src]
	if size <= off {
		return
	}
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	if _, err := f.Seek(off, 0); err != nil {
		return
	}
	rd := bufio.NewReader(f)
	max := int64(65536)
	remain := size - off
	if remain > max {
		if _, err := f.Seek(size-max, 0); err != nil {
			return
		}
		off = size - max
	}
	// Cap emissions per tick so the GUI is not flooded (fyne.Do + full log SetText).
	const maxEmitPerTick = 120
	emitted := 0
	for {
		line, err := rd.ReadString('\n')
		line = strings.TrimSpace(line)
		if line != "" && emitted < maxEmitPerTick {
			m.emitLogFileLine(src, line)
			emitted++
		}
		if err != nil {
			break
		}
	}
	cursors[src] = size
}

func (m *Manager) emitLogFileLine(src, raw string) {
	if src == "activity" {
		lvl := "INFO"
		if strings.Contains(raw, "ERROR") || strings.Contains(raw, "Traceback") {
			lvl = "ERROR"
		} else if strings.Contains(raw, "WARN") {
			lvl = "WARN"
		}
		t := SanitizeLogText(raw)
		if len(t) > 200 {
			t = t[:200]
		}
		m.emit(src, lvl, t)
		return
	}
	lvl := "INFO"
	var obj map[string]interface{}
	if json.Unmarshal([]byte(raw), &obj) == nil {
		if lv, ok := obj["level"].(string); ok {
			switch strings.ToUpper(lv) {
			case "ERROR", "CRITICAL":
				lvl = "ERROR"
			case "WARN", "WARNING":
				lvl = "WARN"
			}
		}
		msg := ""
		for _, k := range []string{"message", "msg", "event", "content"} {
			if v, ok := obj[k]; ok {
				msg = toStr(v)
				if msg != "" {
					break
				}
			}
		}
		if msg == "" {
			msg = raw
		}
		msg = SanitizeLogText(msg)
		if len(msg) > 200 {
			msg = msg[:200]
		}
		m.emit(src, lvl, msg)
		return
	}
	raw = SanitizeLogText(raw)
	if len(raw) > 200 {
		raw = raw[:200]
	}
	m.emit(src, "INFO", raw)
}

func toStr(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	default:
		return ""
	}
}
