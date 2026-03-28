//go:build cgo

package gui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"sancta-launcher/internal/cli"
	"sancta-launcher/internal/core"
)

func init() { runImpl = runFyne }

// runFyne opens the Fyne GUI with Termux-style terminal aesthetic.
func runFyne(ctx context.Context, m *core.Manager) {
	a := app.NewWithID("io.sancta.launcher")
	a.Settings().SetTheme(&termuxTheme{})
	w := a.NewWindow("sancta ~ terminal")
	w.Resize(fyne.NewSize(1100, 720))

	// ── Log buffer ──────────────────────────────────────────
	logBuf := strings.Builder{}
	const maxLog = 120_000
	appendLog := func(s string) {
		logBuf.WriteString(s)
		if logBuf.Len() > maxLog {
			t := logBuf.String()
			logBuf.Reset()
			logBuf.WriteString(t[len(t)-maxLog/2:])
		}
	}

	// ── Live log widget ─────────────────────────────────────
	liveLog := widget.NewMultiLineEntry()
	liveLog.Wrapping = fyne.TextWrapWord
	liveLog.TextStyle = fyne.TextStyle{Monospace: true}
	liveLog.SetText("$ tail -f sancta.log\n# streaming subprocess + project logs\n# use filter to isolate source\n\n")

	// ── CLI transcript ──────────────────────────────────────
	cliOutLbl := widget.NewLabel("")
	cliOutLbl.Wrapping = fyne.TextWrapWord
	cliOutLbl.TextStyle = fyne.TextStyle{Monospace: true}
	cliOutScroll := container.NewScroll(cliOutLbl)
	cliOutText := "Welcome to Sancta Terminal!\n\n  type 'help' for available commands\n\n"
	cliOutLbl.SetText(cliOutText)

	// ── Filter ──────────────────────────────────────────────
	filter := widget.NewSelect([]string{
		"ALL", "SANCTA", "SIEM", "OLLAMA", "CURIOSITY", "PHENOMENOLOGY", "LAUNCHER",
		"SECURITY", "REDTEAM", "BEHAVIORAL", "ACTIVITY",
	}, nil)
	filter.SetSelected("ALL")

	filteredLogText := func() string {
		t := logBuf.String()
		if sel := filter.Selected; sel != "" && sel != "ALL" {
			var lines []string
			up := strings.ToUpper(sel)
			for _, ln := range strings.Split(t, "\n") {
				if strings.Contains(strings.ToUpper(ln), "["+up+"]") ||
					strings.Contains(strings.ToUpper(ln), up) {
					lines = append(lines, ln)
				}
			}
			t = strings.Join(lines, "\n")
		}
		return t
	}

	// Read-only behavior: revert user edits in the log view.
	var liveLogMute bool
	refreshLive := func() {
		liveLogMute = true
		liveLog.SetText(filteredLogText())
		liveLogMute = false
	}
	liveLog.OnChanged = func(_ string) {
		if liveLogMute {
			return
		}
		liveLogMute = true
		liveLog.SetText(filteredLogText())
		liveLogMute = false
	}

	// ── Log batching goroutine ──────────────────────────────
	logCh := m.SubscribeLog(2048)
	go func() {
		const (
			flushEvery   = 200 * time.Millisecond
			flushAtLines = 80
		)
		tick := time.NewTicker(flushEvery)
		defer tick.Stop()
		batch := make([]string, 0, flushAtLines)
		flush := func(lines []string) {
			if len(lines) == 0 {
				return
			}
			fyne.Do(func() {
				for _, s := range lines {
					appendLog(s)
				}
				refreshLive()
			})
		}
		for {
			select {
			case <-ctx.Done():
				return
			case line := <-logCh:
				ts := time.Now().Format("15:04:05")
				s := fmt.Sprintf("%s [%s] %s | %s\n", ts, strings.ToUpper(line.Src), line.Level, line.Text)
				batch = append(batch, s)
				if len(batch) >= flushAtLines {
					flush(batch)
					batch = make([]string, 0, flushAtLines)
				}
			case <-tick.C:
				if len(batch) == 0 {
					continue
				}
				flush(batch)
				batch = make([]string, 0, flushAtLines)
			}
		}
	}()

	// ── Helper constructors ─────────────────────────────────
	mono := func(s string) *widget.Label {
		l := widget.NewLabel(s)
		l.TextStyle = fyne.TextStyle{Monospace: true}
		return l
	}
	monoBold := func(s string) *widget.Label {
		l := widget.NewLabel(s)
		l.TextStyle = fyne.TextStyle{Monospace: true, Bold: true}
		return l
	}

	// ════════════════════════════════════════════════════════
	// ── Services tab ───────────────────────────────────────
	// ════════════════════════════════════════════════════════

	// Banner
	banner := monoBold("$ sancta --control-center")
	title := monoBold("SANCTA CONTROL CENTER v1.0")
	separator := mono("────────────────────────────────────────")
	sysInfo := mono(fmt.Sprintf("  root    : %s\n  backend : %s", m.Paths.Root, m.Paths.Backend))

	// Status labels (● running / ○ stopped)
	statusO := mono("○ stopped")
	statusS := mono("○ stopped")
	statusA := mono("○ stopped")
	statusSG := mono("○ stopped")
	statusSGT := mono("○ stopped")
	statusC := mono("○ stopped")
	statusP := mono("○ stopped")
	sangptSummary := mono("  engine: probing...")
	sangptSummary.Wrapping = fyne.TextWrapWord

	// Service buttons — compact terminal style
	btnOllamaS := widget.NewButton("start", func() { go m.StartOllama() })
	btnOllamaX := widget.NewButton("stop", func() { go m.StopOllama() })
	btnSiemS := widget.NewButton("start", func() { go m.StartSIEM() })
	btnSiemX := widget.NewButton("stop", func() { m.StopOne("siem") })
	btnSanctaS := widget.NewButton("start", func() { go m.StartSancta() })
	btnSanctaX := widget.NewButton("stop", func() { m.StopOne("sancta") })
	btnSangptS := widget.NewButton("start", func() { go m.StartSangptCLI() })
	btnSangptX := widget.NewButton("stop", func() { m.StopOne("sangpt") })
	btnSangptTrainS := widget.NewButton("train", func() { go m.StartSangptTrain() })
	btnSangptTrainX := widget.NewButton("stop", func() { m.StopOne("sangpt_train") })
	btnCur := widget.NewButton("start", func() { go m.ToggleCuriosity() })
	btnPhen := widget.NewButton("start", func() { go m.TogglePhenomenology() })
	btnAll := widget.NewButton("start-all", func() { go m.StartAll(true) })
	btnStop := widget.NewButton("stop-all", func() { go m.StopAll() })
	btnDash := widget.NewButton("dashboard", func() { _ = core.OpenBrowser(core.SIEMURL) })

	// Service rows: name (fixed width) | status | buttons
	svcRow := func(name string, status *widget.Label, btns ...fyne.CanvasObject) *fyne.Container {
		items := []fyne.CanvasObject{mono(fmt.Sprintf("  %-18s", name)), status}
		items = append(items, btns...)
		return container.NewHBox(items...)
	}

	svcCol := container.NewVBox(
		banner,
		title,
		separator,
		sysInfo,
		widget.NewSeparator(),
		monoBold("──── services ────────────────────────────"),
		widget.NewSeparator(),
		svcRow("ollama", statusO, btnOllamaS, btnOllamaX),
		svcRow("siem", statusS, btnSiemS, btnSiemX),
		svcRow("sancta", statusA, btnSanctaS, btnSanctaX),
		svcRow("sangpt-cli", statusSG, btnSangptS, btnSangptX),
		svcRow("sangpt-train", statusSGT, btnSangptTrainS, btnSangptTrainX),
		svcRow("curiosity", statusC, btnCur),
		svcRow("phenomenology", statusP, btnPhen),
		widget.NewSeparator(),
		monoBold("──── sangpt engine ───────────────────────"),
		sangptSummary,
		widget.NewSeparator(),
		monoBold("──── actions ─────────────────────────────"),
		container.NewHBox(btnAll, btnStop, btnDash),
	)
	svcScroll := container.NewScroll(svcCol)

	// ── Status polling goroutine ────────────────────────────
	go func() {
		t := time.NewTicker(500 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				o, s, _ := m.NetStatus()
				rs := m.RunningServices()
				sg := m.SangptStatus()
				fyne.Do(func() {
					if o {
						statusO.SetText("● running")
					} else {
						statusO.SetText("○ stopped")
					}
					if s {
						statusS.SetText("● running")
					} else {
						statusS.SetText("○ stopped")
					}
					if rs["sancta"] {
						statusA.SetText("● running")
					} else {
						statusA.SetText("○ stopped")
					}
					if rs["sangpt"] {
						statusSG.SetText("● running")
					} else {
						statusSG.SetText("○ stopped")
					}
					if rs["sangpt_train"] {
						statusSGT.SetText("● running")
					} else {
						statusSGT.SetText("○ stopped")
					}
					if sg.Backend != "" {
						sangptSummary.SetText("  " + m.SangptSummary())
					} else {
						sangptSummary.SetText("  engine: unavailable")
					}
					if rs["curiosity"] {
						statusC.SetText("● running")
						btnCur.SetText("stop")
					} else {
						statusC.SetText("○ stopped")
						btnCur.SetText("start")
					}
					if rs["phenomenology"] {
						statusP.SetText("● running")
						btnPhen.SetText("stop")
					} else {
						statusP.SetText("○ stopped")
						btnPhen.SetText("start")
					}
				})
			}
		}
	}()

	// ════════════════════════════════════════════════════════
	// ── Live log tab ───────────────────────────────────────
	// ════════════════════════════════════════════════════════
	filter.OnChanged = func(_ string) { refreshLive() }
	btnClr := widget.NewButton("clr", func() {
		logBuf.Reset()
		liveLogMute = true
		liveLog.SetText("")
		liveLogMute = false
	})
	logTitle := monoBold("$ tail -f // live log")
	logHeader := container.NewHBox(logTitle, filter, btnClr)
	logTab := container.NewBorder(logHeader, nil, nil, nil, liveLog)

	// ════════════════════════════════════════════════════════
	// ── Shell tab (embedded REPL) ──────────────────────────
	// ════════════════════════════════════════════════════════
	cliIn := widget.NewEntry()
	cliIn.SetPlaceHolder("sancta $ help | start [svc] | stop [svc] | status | curiosity | phenomenology | dashboard")
	cliIn.OnSubmitted = func(s string) {
		cliOutText = cliOutText + "sancta $ " + s + "\n" + cli.ExecOne(m, s)
		cliOutLbl.SetText(cliOutText)
		cliIn.SetText("")
	}
	cliBox := container.NewBorder(nil, cliIn, nil, nil, cliOutScroll)
	cliHeader := monoBold("$ sancta-shell // type commands below (enter to execute)")
	cliHeader.Wrapping = fyne.TextWrapWord
	cliTab := container.NewBorder(cliHeader, nil, nil, nil, cliBox)

	// ════════════════════════════════════════════════════════
	// ── Tab assembly ───────────────────────────────────────
	// ════════════════════════════════════════════════════════
	cliTabItem := container.NewTabItem("shell", cliTab)
	tabs := container.NewAppTabs(
		container.NewTabItem("services", svcScroll),
		container.NewTabItem("logs", logTab),
		cliTabItem,
	)
	// Focus the entry after the shell tab is shown.
	tabs.OnSelected = func(ti *container.TabItem) {
		if ti != cliTabItem {
			return
		}
		go func() {
			time.Sleep(80 * time.Millisecond)
			fyne.Do(func() {
				if c := w.Canvas(); c != nil {
					c.Focus(cliIn)
				}
			})
		}()
	}
	w.SetContent(tabs)

	w.SetOnClosed(func() {
		m.Emit("launcher", "INFO", "window closed — stopping tracked processes…")
		m.StopAll()
	})

	w.ShowAndRun()
}
