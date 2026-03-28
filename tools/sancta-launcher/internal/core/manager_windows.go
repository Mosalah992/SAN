//go:build windows

package core

import "syscall"

func sysProcAttrHideWindow() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{HideWindow: true}
}

func sysProcAttrNewConsole() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: 0x00000010}
}
