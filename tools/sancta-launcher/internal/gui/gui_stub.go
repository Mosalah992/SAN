//go:build !cgo

package gui

import (
	"context"
	"fmt"
	"os"

	"sancta-launcher/internal/core"
)

func init() { runImpl = runNoCGO }

// runNoCGO reports that the Fyne GUI needs CGO (see README). CLI: sancta-launcher cli
func runNoCGO(_ context.Context, _ *core.Manager) {
	fmt.Fprintln(os.Stderr, "sancta-launcher: GUI requires CGO (C compiler on PATH). Install gcc/MSYS2 per README, or run: sancta-launcher cli")
	os.Exit(1)
}
