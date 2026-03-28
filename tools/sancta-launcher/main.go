// Sancta Control Center — Go port of backend/sancta_launcher.py
//
//   go run .                    GUI (Fyne) + Services / Live log / CLI tabs
//   go run . cli                Terminal REPL
//   go run . start              Start all + stream logs until Ctrl+C
//   go run . status             Print status and exit
//   go run . run ollama|siem|sancta|sangpt|sangpt-train|curiosity|phenomenology
//
// Build (Windows, requires CGO + gcc for Fyne):
//   cd tools/sancta-launcher && go build -ldflags "-s -w -H windowsgui" -o sancta-launcher.exe .
//   (-H windowsgui avoids a background CMD when starting the .exe from Explorer.)
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"sancta-launcher/internal/cli"
	"sancta-launcher/internal/core"
	"sancta-launcher/internal/gui"
)

func main() {
	paths := core.ResolvePaths()
	sanctaPy := filepath.Join(paths.Backend, "sancta.py")
	if _, err := os.Stat(sanctaPy); err != nil {
		fmt.Fprintf(os.Stderr, "Cannot find backend/sancta.py at %q — run from repo root or set working directory.\n", sanctaPy)
		os.Exit(1)
	}

	m := core.NewManager(paths)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go m.RunNetChecker(ctx, 3*time.Second)
	go m.RunLogTail(ctx)

	args := os.Args[1:]
	if len(args) == 0 {
		gui.Run(ctx, m)
		return
	}

	switch args[0] {
	case "gui":
		gui.Run(ctx, m)
	case "cli":
		cli.RunTerminal(ctx, m)
	case "start":
		runStartAttach(ctx, m)
	case "status":
		printStatus(m)
	case "run":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: sancta-launcher run <ollama|siem|sancta|sangpt|sangpt-train|curiosity|phenomenology>")
			os.Exit(1)
		}
		m.StartOne(args[1])
		waitInterrupt()
	case "-h", "--help", "help":
		printHelp()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q — try: gui | cli | start | status | run <svc>\n", args[0])
		os.Exit(1)
	}
}

func runStartAttach(_ context.Context, m *core.Manager) {
	ch := m.SubscribeLog(256)
	go func() {
		for line := range ch {
			fmt.Printf("[%s] %s | %s\n", line.Src, line.Level, line.Text)
		}
	}()
	m.StartAll(true)
	waitInterrupt()
	m.StopAll()
}

func waitInterrupt() {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	<-sig
}

func printStatus(m *core.Manager) {
	m.RefreshStatus()
	o, s, model, sp, sgp, sgt, cp, pp := m.StatusSnapshot()
	fmt.Printf("Ollama:     %v  model_tag=%q\n", o, model)
	fmt.Printf("SIEM:       %v  %s\n", s, core.SIEMURL)
	fmt.Printf("Sancta pid: %d\n", sp)
	fmt.Printf("Sangpt CLI pid: %d\n", sgp)
	fmt.Printf("Sangpt Train pid: %d\n", sgt)
	fmt.Printf("Sangpt status: %s\n", m.SangptSummary())
	fmt.Printf("Curiosity:  %d\n", cp)
	fmt.Printf("Phenomenology: %d\n", pp)
	fmt.Printf("Root: %s\nBackend: %s\n", m.Paths.Root, m.Paths.Backend)
}

func printHelp() {
	fmt.Print(`sancta-launcher — Go control center (Fyne GUI + CLI)

  (no args)   Launch GUI: Services | Live log | CLI tab
  gui         Same as default
  cli         Interactive terminal (stdin)
  start       Start Ollama→SIEM→Sancta, open browser, log until Ctrl+C
  status      Print health snapshot
  run <svc>   Start one service then wait for Ctrl+C

Services: ollama, siem, sancta, sangpt, sangpt-train, curiosity, phenomenology

Build:  go build -ldflags "-s -w -H windowsgui" -o sancta-launcher.exe .
Needs:  C compiler (CGO) for Fyne on Windows — see README.md in this folder.
`)
}
