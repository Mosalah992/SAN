package core

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Paths holds resolved project root and backend directory (same rules as sancta_launcher.py).
type Paths struct {
	Root    string
	Backend string
	SangptCLI string
	SangptTrain string
}

// ResolvePaths finds ROOT and BACKEND (directory containing sancta.py).
func ResolvePaths() Paths {
	if _, g, ok := findRepoFromWD(); ok {
		return makePaths(g.parent, g.backend)
	}
	wd, _ := os.Getwd()
	if p := tryPathsFromDir(wd); p.Root != "" {
		return p
	}
	exe, err := os.Executable()
	if err == nil {
		exe, _ = filepath.EvalSymlinks(exe)
		exeDir := filepath.Dir(exe)
		if p := tryPathsFromDir(exeDir); p.Root != "" {
			return p
		}
		for _, up := range []string{filepath.Dir(exeDir), filepath.Dir(filepath.Dir(exeDir))} {
			if p := tryPathsFromDir(up); p.Root != "" {
				return p
			}
		}
	}
	return makePaths(wd, filepath.Join(wd, "backend"))
}

type repoGuess struct {
	parent  string
	backend string
}

func findRepoFromWD() (string, repoGuess, bool) {
	wd, err := os.Getwd()
	if err != nil {
		return "", repoGuess{}, false
	}
	dir := wd
	for i := 0; i < 14; i++ {
		backend := filepath.Join(dir, "backend")
		if fileExists(filepath.Join(backend, "sancta.py")) {
			return dir, repoGuess{parent: dir, backend: backend}, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", repoGuess{}, false
}

func tryPathsFromDir(dir string) Paths {
	dir = filepath.Clean(dir)
	if fileExists(filepath.Join(dir, "sancta.py")) && filepath.Base(dir) == "backend" {
		return makePaths(filepath.Dir(dir), dir)
	}
	b := filepath.Join(dir, "backend")
	if fileExists(filepath.Join(b, "sancta.py")) {
		return makePaths(dir, b)
	}
	return Paths{}
}

func makePaths(root, backend string) Paths {
	return Paths{
		Root:        root,
		Backend:     backend,
		SangptCLI:   filepath.Join(backend, "run_sangpt_cli.py"),
		SangptTrain: filepath.Join(backend, "run_sancta_gpt_training.py"),
	}
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// PythonExe returns interpreter to launch Sancta / uvicorn.
func PythonExe() string {
	if runtime.GOOS == "windows" {
		for _, name := range []string{"python", "python3", "py"} {
			if p, err := exec.LookPath(name); err == nil {
				return p
			}
		}
		return "python"
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	if p, err := exec.LookPath("python"); err == nil {
		return p
	}
	return "python3"
}

// FindOllamaExe resolves ollama binary (OLLAMA_EXE, PATH, common Windows paths).
func FindOllamaExe() string {
	if v := strings.TrimSpace(os.Getenv("OLLAMA_EXE")); v != "" {
		if fileExists(v) {
			return v
		}
	}
	if p, err := exec.LookPath("ollama"); err == nil {
		return p
	}
	if runtime.GOOS == "windows" {
		local := os.Getenv("LOCALAPPDATA")
		candidates := []string{
			filepath.Join(local, "Programs", "Ollama", "ollama.exe"),
			filepath.Join(os.Getenv("ProgramFiles"), "Ollama", "ollama.exe"),
		}
		for _, c := range candidates {
			if fileExists(c) {
				return c
			}
		}
	}
	return "ollama"
}
