package gui

import (
	"context"

	"sancta-launcher/internal/core"
)

// runImpl is registered from gui_fyne.go (cgo) or gui_stub.go (!cgo).
var runImpl func(context.Context, *core.Manager)

// Run opens the GUI when CGO is available, otherwise prints a hint and exits.
func Run(ctx context.Context, m *core.Manager) {
	runImpl(ctx, m)
}
