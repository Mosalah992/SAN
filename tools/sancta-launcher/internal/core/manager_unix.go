//go:build !windows

package core

import "syscall"

func sysProcAttrHideWindow() *syscall.SysProcAttr {
	return nil
}

func sysProcAttrNewConsole() *syscall.SysProcAttr {
	return nil
}
