package cli

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"sancta-launcher/internal/core"
)

// RunTerminal interactive REPL (stdin/stdout).
func RunTerminal(ctx context.Context, m *core.Manager) {
	ch := m.SubscribeLog(512)
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case line := <-ch:
				fmt.Printf("[%s] %s | %s\n", strings.ToUpper(line.Src), line.Level, line.Text)
			}
		}
	}()

	sc := bufio.NewScanner(os.Stdin)
	fmt.Println("⬡ Sancta Control — CLI mode. Type 'help'. Ctrl+C or 'exit' to quit.")
	for {
		fmt.Print("sancta> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if handleLine(m, line) {
			break
		}
	}
}

// ExecOne runs a single command line (for GUI CLI tab). Returns multi-line output text.
func ExecOne(m *core.Manager, line string) string {
	line = strings.TrimSpace(line)
	if line == "" {
		return ""
	}
	var b strings.Builder
	parts := strings.Fields(line)
	cmd := strings.ToLower(parts[0])
	arg := ""
	if len(parts) > 1 {
		arg = strings.TrimSpace(line[len(parts[0]):])
	}
	switch cmd {
	case "help", "?":
		b.WriteString(helpText())
	case "start":
		if arg != "" {
			go m.StartOne(arg)
			b.WriteString("starting " + arg + "…\n")
		} else {
			go m.StartAll(false)
			b.WriteString("starting full stack…\n")
		}
	case "stop":
		if arg != "" {
			m.StopOne(arg)
			b.WriteString("stopped " + arg + "\n")
		} else {
			m.StopAll()
			b.WriteString("stopped all\n")
		}
	case "status", "st":
		m.RefreshStatus()
		o, s, model, sp, sgp, sgt, cp, pp := m.StatusSnapshot()
		b.WriteString(fmt.Sprintf("Ollama: %v model=%q\n", o, model))
		b.WriteString(fmt.Sprintf("SIEM:   %v\n", s))
		b.WriteString(fmt.Sprintf("Sancta: pid=%d\n", sp))
		b.WriteString(fmt.Sprintf("Sangpt: pid=%d\n", sgp))
		b.WriteString(fmt.Sprintf("Sangpt train: pid=%d\n", sgt))
		b.WriteString(fmt.Sprintf("Sangpt status: %s\n", m.SangptSummary()))
		b.WriteString(fmt.Sprintf("Curiosity: pid=%d\n", cp))
		b.WriteString(fmt.Sprintf("Phenomenology: pid=%d\n", pp))
	case "curiosity":
		go m.ToggleCuriosity()
		b.WriteString("curiosity toggled\n")
	case "phenomenology", "phenom":
		go m.TogglePhenomenology()
		b.WriteString("phenomenology toggled\n")
	case "dashboard", "dash":
		_ = openURL(core.SIEMURL)
		b.WriteString("opened dashboard\n")
	case "clear", "cls":
		b.WriteString("(clear — use GUI CLR or terminal cls locally)\n")
	default:
		b.WriteString("unknown command — type help\n")
	}
	return b.String()
}

func handleLine(m *core.Manager, line string) bool {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return false
	}
	cmd := strings.ToLower(parts[0])
	arg := ""
	if len(parts) > 1 {
		arg = strings.TrimSpace(line[len(parts[0]):])
	}
	switch cmd {
	case "exit", "quit", "q":
		m.Emit("launcher", "INFO", "Shutting down…")
		m.StopAll()
		time.Sleep(time.Second)
		return true
	case "help", "?":
		fmt.Print(helpText())
	case "start":
		if arg != "" {
			go m.StartOne(arg)
		} else {
			go m.StartAll(true)
		}
	case "stop":
		if arg != "" {
			m.StopOne(arg)
		} else {
			m.StopAll()
		}
	case "status", "st":
		m.RefreshStatus()
		o, s, model, sp, sgp, sgt, cp, pp := m.StatusSnapshot()
		fmt.Printf("  Ollama: %v  %s\n", o, model)
		fmt.Printf("  SIEM:   %v\n", s)
		fmt.Printf("  Sancta pid: %d  Sangpt pid: %d  Sangpt train pid: %d  Curiosity pid: %d  Phenom pid: %d\n", sp, sgp, sgt, cp, pp)
		fmt.Printf("  Sangpt status: %s\n", m.SangptSummary())
	case "curiosity":
		go m.ToggleCuriosity()
	case "phenomenology", "phenom":
		go m.TogglePhenomenology()
	case "dashboard", "dash":
		_ = openURL(core.SIEMURL)
	case "clear", "cls":
		// no-op in portable CLI
	case "restart":
		svc := arg
		if svc == "" {
			svc = "sancta"
		}
		m.StopOne(svc)
		time.Sleep(2 * time.Second)
		go m.StartOne(svc)
	default:
		fmt.Println("  unknown command — type help")
	}
	return false
}

func helpText() string {
	return `
  start              Start Ollama → SIEM → Sancta (browser open in terminal mode only via 'start' from main)
  start <service>    ollama | siem | sancta | sangpt | sangpt-train | curiosity | phenomenology
  stop               Stop all (incl. Ollama listener)
  stop <service>
  status             Process / health snapshot
  curiosity          Toggle curiosity run
  phenomenology      Toggle phenomenology battery
  dashboard          Open SIEM in browser
  help
  exit | quit
`
}

func openURL(url string) error {
	return core.OpenBrowser(url)
}
