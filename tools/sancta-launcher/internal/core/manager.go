package core

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	OllamaURL = "http://127.0.0.1:11434"
	SIEMURL   = "http://127.0.0.1:8787"
)

// LogLine is pushed to subscribers (GUI / CLI).
type LogLine struct {
	Src   string
	Level string
	Text  string
}

type SangptStatus struct {
	Backend          string   `json:"backend"`
	Initialized      bool     `json:"initialized"`
	Ready            bool     `json:"ready"`
	CheckpointExists bool     `json:"checkpoint_exists"`
	TrainingMode     string   `json:"training_mode"`
	CorpusSize       int      `json:"corpus_size"`
	Step             int      `json:"step"`
	VocabSize        int      `json:"vocab_size"`
	NumParams        int      `json:"num_params"`
	LastLoss         *float64 `json:"last_loss"`
}

// Manager orchestrates subprocesses (Python launcher parity).
type Manager struct {
	Paths Paths

	mu       sync.Mutex
	procs    map[string]*exec.Cmd
	logSubs  []chan LogLine
	netMu    sync.RWMutex
	ollamaOK bool
	siemOK   bool
	model    string
	sangptMu sync.RWMutex
	sangpt   SangptStatus

	curiosityRunning bool
	restartCounts    map[string]int
}

func NewManager(p Paths) *Manager {
	return &Manager{
		Paths:         p,
		procs:         make(map[string]*exec.Cmd),
		restartCounts: make(map[string]int),
	}
}

// SubscribeLog returns a channel; caller should drain or goroutine will block producer (buffered 256).
func (m *Manager) SubscribeLog(buf int) chan LogLine {
	if buf <= 0 {
		buf = 256
	}
	ch := make(chan LogLine, buf)
	m.mu.Lock()
	m.logSubs = append(m.logSubs, ch)
	m.mu.Unlock()
	return ch
}

// Emit pushes a log line (exported for CLI glue).
func (m *Manager) Emit(src, level, text string) {
	m.emit(src, level, text)
}

func (m *Manager) emit(src, level, text string) {
	line := LogLine{Src: src, Level: level, Text: text}
	m.mu.Lock()
	subs := append([]chan LogLine(nil), m.logSubs...)
	m.mu.Unlock()
	for _, ch := range subs {
		select {
		case ch <- line:
		default:
			// drop if slow consumer
		}
	}
}

func (m *Manager) NetStatus() (ollama, siem bool, model string) {
	m.netMu.RLock()
	defer m.netMu.RUnlock()
	return m.ollamaOK, m.siemOK, m.model
}

func (m *Manager) SangptStatus() SangptStatus {
	m.sangptMu.RLock()
	defer m.sangptMu.RUnlock()
	return m.sangpt
}

func (m *Manager) SangptSummary() string {
	s := m.SangptStatus()
	if s.Backend == "" {
		return "Sangpt status unavailable"
	}
	loss := "n/a"
	if s.LastLoss != nil {
		loss = fmt.Sprintf("%.4f", *s.LastLoss)
	}
	ready := "warming"
	if s.Ready {
		ready = "ready"
	} else if s.Initialized {
		ready = "initialized"
	}
	return fmt.Sprintf(
		"backend=%s | %s | corpus=%d | step=%d | loss=%s | ckpt=%v | mode=%s",
		s.Backend, ready, s.CorpusSize, s.Step, loss, s.CheckpointExists, s.TrainingMode,
	)
}

func (m *Manager) refreshHealthSnapshot() {
	okO := m.pingOllama()
	okS := m.pingSIEM()
	model := ""
	if okO {
		model = m.fetchOllamaModelTag()
	}
	sangpt := m.querySangptStatus()
	m.netMu.Lock()
	m.ollamaOK, m.siemOK, m.model = okO, okS, model
	m.netMu.Unlock()
	m.sangptMu.Lock()
	m.sangpt = sangpt
	m.sangptMu.Unlock()
}

// RefreshStatus forces an immediate probe so CLI/GUI can show current state
// without waiting for the periodic ticker.
func (m *Manager) RefreshStatus() {
	m.refreshHealthSnapshot()
}

// RunNetChecker updates ollama/siem flags every d.
func (m *Manager) RunNetChecker(ctx context.Context, d time.Duration) {
	m.refreshHealthSnapshot()
	t := time.NewTicker(d)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			m.refreshHealthSnapshot()
		}
	}
}

func (m *Manager) querySangptStatus() SangptStatus {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	py := PythonExe()
	   cmd := exec.CommandContext(
		   ctx,
		   py,
		   "-c",
		   "import json, sancta_gpt; print(json.dumps(sancta_gpt.status()))",
	   )
	   cmd.Dir = m.Paths.Root
	   hideWindow(cmd)
	   out, err := cmd.Output()
	if err != nil {
		return SangptStatus{}
	}
	var st SangptStatus
	if err := json.Unmarshal(out, &st); err != nil {
		return SangptStatus{}
	}
	return st
}

func (m *Manager) pingOllama() bool {
	c := &http.Client{Timeout: 1500 * time.Millisecond}
	r, err := c.Get(OllamaURL + "/api/version")
	if err != nil {
		return false
	}
	r.Body.Close()
	return r.StatusCode == 200
}

func (m *Manager) pingSIEM() bool {
	c := &http.Client{Timeout: 1500 * time.Millisecond}
	r, err := c.Get(SIEMURL)
	if err != nil {
		return false
	}
	r.Body.Close()
	return r.StatusCode < 500
}

func (m *Manager) fetchOllamaModelTag() string {
	c := &http.Client{Timeout: 1500 * time.Millisecond}
	r, err := c.Get(OllamaURL + "/api/tags")
	if err != nil || r.StatusCode != 200 {
		return ""
	}
	defer r.Body.Close()
	// minimal parse: look for "name":"...llama3.2..."
	b := make([]byte, 65536)
	n, _ := r.Body.Read(b)
	s := string(b[:n])
	pref := os.Getenv("LOCAL_MODEL")
	if pref == "" {
		pref = "llama3.2"
	}
	if idx := strings.Index(s, pref); idx >= 0 {
		// best-effort: return pref
		return pref
	}
	return ""
}

func agentPIDPath(root string) string {
	return filepath.Join(root, ".agent.pid")
}

func (m *Manager) writeAgentPID(pid int) {
	_ = os.WriteFile(agentPIDPath(m.Paths.Root), []byte(strconv.Itoa(pid)), 0644)
}

func (m *Manager) clearAgentPIDIfMatches(pid int) {
	p := agentPIDPath(m.Paths.Root)
	b, err := os.ReadFile(p)
	if err != nil {
		return
	}
	if strings.TrimSpace(string(b)) == strconv.Itoa(pid) {
		_ = os.Remove(p)
	}
}

func hideWindow(cmd *exec.Cmd) {
	if a := sysProcAttrHideWindow(); a != nil {
		cmd.SysProcAttr = a
	}
}

// StartOllama spawns `ollama serve` if API not up.
func (m *Manager) StartOllama() bool {
	if m.pingOllama() {
		m.emit("ollama", "INFO", "Already running — connecting")
		return true
	}
	m.emit("ollama", "INFO", "Starting Ollama...")
	ex := FindOllamaExe()
	cmd := exec.Command(ex, "serve")
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	hideWindow(cmd)
	if err := cmd.Start(); err != nil {
		m.emit("ollama", "ERROR", err.Error())
		return false
	}
	m.mu.Lock()
	m.procs["ollama"] = cmd
	m.mu.Unlock()
	for i := 0; i < 15; i++ {
		time.Sleep(time.Second)
		if m.pingOllama() {
			m.emit("ollama", "OK", "Ollama ready on :11434")
			return true
		}
	}
	m.emit("ollama", "ERROR", "Timeout waiting for Ollama")
	return false
}

// StopOllama stops tracked serve then kills :11434 listener if still up.
func (m *Manager) StopOllama() {
	m.stopProcName("ollama")
	time.Sleep(400 * time.Millisecond)
	if !m.pingOllama() {
		m.emit("ollama", "INFO", "Ollama stopped")
		return
	}
	pid := pidListeningPort(11434)
	if pid > 0 {
		m.emit("ollama", "WARN", fmt.Sprintf("Stopping process on :11434 (PID %d)", pid))
		killPIDHard(pid)
		time.Sleep(400 * time.Millisecond)
	}
	if m.pingOllama() {
		m.emit("ollama", "WARN", "Ollama API still up — quit from system tray if needed")
	} else {
		m.emit("ollama", "INFO", "Ollama stopped")
	}
}

// StartSIEM runs uvicorn from repo root.
func (m *Manager) StartSIEM() bool {
	if m.pingSIEM() {
		m.emit("siem", "INFO", "Already running on :8787")
		return true
	}
	py := PythonExe()
	args := []string{"-m", "uvicorn", "backend.siem_server:app", "--host", "127.0.0.1", "--port", "8787"}
	cmd := exec.Command(py, args...)
	cmd.Dir = m.Paths.Root
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1", "SIEM_METRICS_SAFE_MODE=false", "SIEM_WS_SAFE_MODE=false")
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = cmd.Stdout
	hideWindow(cmd)
	if err := cmd.Start(); err != nil {
		m.emit("siem", "ERROR", err.Error())
		return false
	}
	m.mu.Lock()
	m.procs["siem"] = cmd
	m.mu.Unlock()
	m.emit("siem", "OK", fmt.Sprintf("Started (PID %d)", cmd.Process.Pid))
	go func() {
		m.streamProcOutput("siem", stdout)
		_ = cmd.Wait()
		m.mu.Lock()
		delete(m.procs, "siem")
		m.mu.Unlock()
		m.emit("siem", "INFO", "SIEM server exited")
	}()
	return true
}

// StartSancta runs sancta.py with auto-restart on exit (same spirit as Python).
func (m *Manager) StartSancta() bool {
	m.mu.Lock()
	_, busy := m.procs["sancta"]
	m.mu.Unlock()
	if busy {
		m.emit("sancta", "INFO", "Already running")
		return true
	}
	return m.startSanctaOnce(true)
}

func (m *Manager) startSanctaOnce(restart bool) bool {
	py := PythonExe()
	script := filepath.Join(m.Paths.Backend, "sancta.py")
	cmd := exec.Command(py, script)
	cmd.Dir = m.Paths.Backend
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1", "OLLAMA_CONTEXT_LENGTH=8192")
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = cmd.Stdout
	hideWindow(cmd)
	if err := cmd.Start(); err != nil {
		m.emit("sancta", "ERROR", err.Error())
		return false
	}
	m.writeAgentPID(cmd.Process.Pid)
	m.mu.Lock()
	m.procs["sancta"] = cmd
	m.mu.Unlock()
	m.emit("sancta", "OK", fmt.Sprintf("Started (PID %d)", cmd.Process.Pid))
	go m.streamSanctaOutput(cmd, script, restart, stdout)
	return true
}

func (m *Manager) streamProcOutput(name string, r io.Reader) {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := SanitizeLogText(sc.Text())
		if line == "" {
			continue
		}
		lvl := "INFO"
		if strings.Contains(line, "ERROR") || strings.Contains(line, "Traceback") {
			lvl = "ERROR"
		} else if strings.Contains(line, "WARN") {
			lvl = "WARN"
		}
		if len(line) > 200 {
			line = line[:200]
		}
		m.emit(name, lvl, line)
	}
	_ = sc.Err()
}

func (m *Manager) streamSanctaOutput(cmd *exec.Cmd, _ string, restart bool, r io.Reader) {
	m.streamProcOutput("sancta", r)
	err := cmd.Wait()
	pid := 0
	if cmd.Process != nil {
		pid = cmd.Process.Pid
	}
	m.clearAgentPIDIfMatches(pid)
	m.mu.Lock()
	delete(m.procs, "sancta")
	m.mu.Unlock()
	code := 0
	if err != nil {
		code = 1
	}
	if code != 0 {
		m.emit("sancta", "WARN", fmt.Sprintf("Exited (code %d)", code))
	} else {
		m.emit("sancta", "INFO", "Exited (code 0)")
	}
	m.mu.Lock()
	cr := m.curiosityRunning
	m.mu.Unlock()
	if restart && !cr {
		m.restartCounts["sancta"]++
		m.emit("sancta", "WARN", "Auto-restarting in 10s...")
		time.Sleep(10 * time.Second)
		m.mu.Lock()
		_, still := m.procs["sancta"]
		m.mu.Unlock()
		if !still {
			m.startSanctaOnce(true)
		}
	}
}

// StartCuriosity runs sancta.py --curiosity-run from backend dir.
func (m *Manager) StartCuriosity() bool {
	if !m.pingOllama() {
		m.emit("curiosity", "ERROR", "Ollama must be running first")
		return false
	}
	m.curiosityRunning = true
	py := PythonExe()
	script := filepath.Join(m.Paths.Backend, "sancta.py")
	cmd := exec.Command(py, script, "--curiosity-run")
	cmd.Dir = m.Paths.Backend
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = cmd.Stdout
	hideWindow(cmd)
	if err := cmd.Start(); err != nil {
		m.curiosityRunning = false
		m.emit("curiosity", "ERROR", err.Error())
		return false
	}
	m.mu.Lock()
	m.procs["curiosity"] = cmd
	m.mu.Unlock()
	m.emit("curiosity", "OK", fmt.Sprintf("Started (PID %d)", cmd.Process.Pid))
	go func() {
		m.streamProcOutput("curiosity", stdout)
		_ = cmd.Wait()
		m.mu.Lock()
		m.curiosityRunning = false
		delete(m.procs, "curiosity")
		m.mu.Unlock()
		m.emit("curiosity", "INFO", "Curiosity run finished")
	}()
	return true
}

// StartPhenomenology runs sancta.py --phenomenology-battery.
func (m *Manager) StartPhenomenology() bool {
	py := PythonExe()
	script := filepath.Join(m.Paths.Backend, "sancta.py")
	cmd := exec.Command(py, script, "--phenomenology-battery")
	cmd.Dir = m.Paths.Backend
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = cmd.Stdout
	hideWindow(cmd)
	if err := cmd.Start(); err != nil {
		m.emit("phenomenology", "ERROR", err.Error())
		return false
	}
	m.mu.Lock()
	m.procs["phenomenology"] = cmd
	m.mu.Unlock()
	m.emit("phenomenology", "OK", fmt.Sprintf("Started (PID %d)", cmd.Process.Pid))
	go func() {
		m.streamProcOutput("phenomenology", stdout)
		_ = cmd.Wait()
		m.mu.Lock()
		delete(m.procs, "phenomenology")
		m.mu.Unlock()
	}()
	return true
}

// StartSangptCLI launches the merged Sangpt terminal wrapper in a new PowerShell window.
func (m *Manager) StartSangptCLI() bool {
	m.mu.Lock()
	_, busy := m.procs["sangpt"]
	m.mu.Unlock()
	if busy {
		m.emit("sangpt", "INFO", "Already running")
		return true
	}
	   py := PythonExe()
	   script := m.Paths.SangptCLI
	   powershellCmd := []string{
		   "cmd", "/c", "start", "powershell", "-NoExit", "-Command",
		   fmt.Sprintf("cd \"%s\"; & \"%s\" \"%s\"", m.Paths.Root, py, script),
	   }
	   cmd := exec.Command(powershellCmd[0], powershellCmd[1:]...)
	   cmd.Dir = m.Paths.Root
	   cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	   // Add a max restart limit for Sangpt CLI to prevent infinite pop-ups
	   maxRestarts := 5
	   m.restartCounts["sangpt"]++
	   if m.restartCounts["sangpt"] > maxRestarts {
		   m.emit("sangpt", "ERROR", fmt.Sprintf("Max restart attempts (%d) reached. Not restarting.", maxRestarts))
		   return false
	   }
	   if err := cmd.Start(); err != nil {
		   m.emit("sangpt", "ERROR", err.Error())
		   return false
	   }
	   m.mu.Lock()
	   m.procs["sangpt"] = cmd
	   m.mu.Unlock()
	   m.emit("sangpt", "OK", fmt.Sprintf("Opened interactive console (PID %d)", cmd.Process.Pid))
	   go func() {
		   _ = cmd.Wait()
		   m.mu.Lock()
		   delete(m.procs, "sangpt")
		   m.mu.Unlock()
		   m.emit("sangpt", "INFO", "Interactive console exited")
	   }()
	   return true
}

// StartSangptTrain launches a bounded training run for the merged Sangpt engine.
func (m *Manager) StartSangptTrain() bool {
	m.mu.Lock()
	_, busy := m.procs["sangpt_train"]
	m.mu.Unlock()
	if busy {
		m.emit("sangpt_train", "INFO", "Training already running")
		return true
	}
	py := PythonExe()
	cmd := exec.Command(py, m.Paths.SangptTrain, "25")
	cmd.Dir = m.Paths.Root
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = cmd.Stdout
	hideWindow(cmd)
	if err := cmd.Start(); err != nil {
		m.emit("sangpt_train", "ERROR", err.Error())
		return false
	}
	m.mu.Lock()
	m.procs["sangpt_train"] = cmd
	m.mu.Unlock()
	m.emit("sangpt_train", "OK", fmt.Sprintf("Started (PID %d)", cmd.Process.Pid))
	go func() {
		m.streamProcOutput("sangpt_train", stdout)
		_ = cmd.Wait()
		m.mu.Lock()
		delete(m.procs, "sangpt_train")
		m.mu.Unlock()
		m.emit("sangpt_train", "INFO", "Sangpt training run finished")
	}()
	return true
}

func (m *Manager) stopProcName(name string) {
	m.mu.Lock()
	cmd := m.procs[name]
	delete(m.procs, name)
	m.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	if name == "sancta" {
		m.clearAgentPIDIfMatches(pid)
	}
	_ = cmd.Process.Kill()
	_, _ = cmd.Process.Wait()
	m.emit(name, "INFO", "Stopped")
}

// StopAll stops phenomenology, curiosity, sancta, siem, ollama.
func (m *Manager) StopAll() {
	for _, n := range []string{"phenomenology", "curiosity", "sangpt_train", "sangpt", "sancta", "siem"} {
		m.stopProcName(n)
	}
	m.StopOllama()
}

func (m *Manager) StopOne(name string) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "ollama":
		m.StopOllama()
	case "siem":
		m.stopProcName("siem")
	case "sancta":
		m.stopProcName("sancta")
	case "sangpt":
		m.stopProcName("sangpt")
	case "sangpt-train", "sangpt_train":
		m.stopProcName("sangpt_train")
	case "curiosity":
		m.curiosityRunning = false
		m.stopProcName("curiosity")
	case "phenomenology":
		m.stopProcName("phenomenology")
	default:
		m.emit("launcher", "WARN", "unknown service: "+name)
	}
}

// StartOne starts a single service by name.
func (m *Manager) StartOne(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "ollama":
		return m.StartOllama()
	case "siem":
		return m.StartSIEM()
	case "sancta":
		return m.StartSancta()
	case "sangpt":
		return m.StartSangptCLI()
	case "sangpt-train", "sangpt_train":
		return m.StartSangptTrain()
	case "curiosity":
		return m.StartCuriosity()
	case "phenomenology", "phenom":
		return m.StartPhenomenology()
	default:
		m.emit("launcher", "ERROR", "unknown service: "+name)
		return false
	}
}

// StartAll: Ollama → SIEM → Sancta (blocking steps with waits).
func (m *Manager) StartAll(openBrowser bool) {
	m.emit("launcher", "INFO", "─── Starting Sancta stack ───")
	if !m.StartOllama() {
		m.emit("launcher", "ERROR", "Ollama failed — check installation")
		return
	}
	time.Sleep(time.Second)
	if !m.pingSIEM() {
		m.emit("launcher", "INFO", "Starting SIEM server...")
		m.StartSIEM()
		for i := 0; i < 10; i++ {
			time.Sleep(time.Second)
			if m.pingSIEM() {
				m.emit("siem", "OK", "Dashboard ready on :8787")
				break
			}
		}
	} else {
		m.emit("siem", "INFO", "Already running on :8787")
	}
	m.emit("launcher", "INFO", "Starting Sancta agent...")
	m.StartSancta()
	time.Sleep(2 * time.Second)
	if openBrowser {
		_ = OpenBrowser(SIEMURL)
		m.emit("launcher", "INFO", "Opened dashboard in browser")
	}
	m.emit("launcher", "OK", "All services started ✓")
}

// StatusSnapshot for CLI status table.
func (m *Manager) StatusSnapshot() (ollama, siem bool, model string, sanctaPID, sangptPID, sangptTrainPID, curPID, phenPID int) {
	ollama, siem, model = m.NetStatus()
	m.mu.Lock()
	defer m.mu.Unlock()
	if c := m.procs["sancta"]; c != nil && c.Process != nil {
		sanctaPID = c.Process.Pid
	}
	if c := m.procs["sangpt"]; c != nil && c.Process != nil {
		sangptPID = c.Process.Pid
	}
	if c := m.procs["sangpt_train"]; c != nil && c.Process != nil {
		sangptTrainPID = c.Process.Pid
	}
	if c := m.procs["curiosity"]; c != nil && c.Process != nil {
		curPID = c.Process.Pid
	}
	if c := m.procs["phenomenology"]; c != nil && c.Process != nil {
		phenPID = c.Process.Pid
	}
	return
}

func pidListeningPort(port int) int {
	if runtime.GOOS == "windows" {
		out, err := exec.Command("netstat", "-ano").Output()
		if err != nil {
			return 0
		}
		needle := fmt.Sprintf(":%d", port)
		for _, line := range strings.Split(string(out), "\n") {
			u := strings.ToUpper(line)
			if !strings.Contains(u, "LISTENING") || !strings.Contains(line, needle) {
				continue
			}
			f := strings.Fields(line)
			if len(f) == 0 {
				continue
			}
			if pid, err := strconv.Atoi(f[len(f)-1]); err == nil {
				return pid
			}
		}
		return 0
	}
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port)).Output()
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return 0
	}
	pid, err := strconv.Atoi(strings.Fields(s)[0])
	if err != nil {
		return 0
	}
	return pid
}

func killPIDHard(pid int) {
	if pid <= 0 {
		return
	}
	if runtime.GOOS == "windows" {
		exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/F").Run()
		return
	}
	exec.Command("kill", "-9", strconv.Itoa(pid)).Run()
}

// OpenBrowser opens a URL in the default handler (exported).
func OpenBrowser(url string) error {
	switch runtime.GOOS {
	case "windows":
		return exec.Command("cmd", "/c", "start", url).Start()
	case "darwin":
		return exec.Command("open", url).Start()
	default:
		return exec.Command("xdg-open", url).Start()
	}
}

// IsCuriosityRunning helper for UI.
func (m *Manager) IsCuriosityRunning() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	c := m.procs["curiosity"]
	return c != nil && c.Process != nil
}

// ToggleCuriosity start/stop.
func (m *Manager) ToggleCuriosity() {
	if m.IsCuriosityRunning() {
		m.emit("curiosity", "INFO", "Stopping curiosity run...")
		m.curiosityRunning = false
		m.stopProcName("curiosity")
		return
	}
	m.StartCuriosity()
}

// RunningServices returns which named processes are still in the table (best-effort alive).
func (m *Manager) RunningServices() map[string]bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]bool)
	for k, c := range m.procs {
		out[k] = c != nil && c.Process != nil
	}
	return out
}

func (m *Manager) IsPhenomenologyRunning() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	c := m.procs["phenomenology"]
	return c != nil && c.Process != nil
}

func (m *Manager) TogglePhenomenology() {
	if m.IsPhenomenologyRunning() {
		m.emit("phenomenology", "INFO", "Stopping phenomenology battery...")
		m.stopProcName("phenomenology")
		return
	}
	m.emit("phenomenology", "INFO", "─── Running phenomenology attack battery ───")
	m.StartPhenomenology()
}

