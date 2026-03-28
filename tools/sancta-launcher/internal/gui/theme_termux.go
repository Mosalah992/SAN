//go:build cgo

package gui

import (
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"
)

// termuxTheme gives the launcher a Termux-style terminal aesthetic:
// black background, green monospace text, dark-green accents.
type termuxTheme struct{}

var _ fyne.Theme = (*termuxTheme)(nil)

// green helper colors (reused in Color below).
var (
	colBg          = color.NRGBA{0x00, 0x00, 0x00, 0xff} // pure black
	colFg          = color.NRGBA{0x33, 0xff, 0x00, 0xff} // bright green
	colFgDim       = color.NRGBA{0x22, 0x88, 0x22, 0xff} // muted green
	colBtn         = color.NRGBA{0x0a, 0x1e, 0x0a, 0xff} // very dark green
	colBtnDisabled = color.NRGBA{0x12, 0x12, 0x12, 0xff}
	colDisabled    = color.NRGBA{0x33, 0x55, 0x33, 0xff}
	colInput       = color.NRGBA{0x08, 0x08, 0x08, 0xff}
	colInputBorder = color.NRGBA{0x1a, 0x44, 0x1a, 0xff}
	colSelection   = color.NRGBA{0x00, 0x33, 0x00, 0xff}
	colSeparator   = color.NRGBA{0x1a, 0x33, 0x1a, 0xff}
	colHeader      = color.NRGBA{0x05, 0x0f, 0x05, 0xff}
	colHover       = color.NRGBA{0x0a, 0x33, 0x0a, 0xff}
	colPressed     = color.NRGBA{0x00, 0x55, 0x00, 0xff}
	colFocus       = color.NRGBA{0x00, 0xcc, 0x00, 0xff}
	colCyan        = color.NRGBA{0x00, 0xe5, 0xff, 0xff}
	colRed         = color.NRGBA{0xff, 0x33, 0x33, 0xff}
	colAmber       = color.NRGBA{0xff, 0xaa, 0x00, 0xff}
	colGreen       = color.NRGBA{0x00, 0xff, 0x66, 0xff}
	colScroll      = color.NRGBA{0x14, 0x3c, 0x14, 0xcc}
	colShadow      = color.NRGBA{0x00, 0x00, 0x00, 0x78}
	colOverlay     = color.NRGBA{0x05, 0x0a, 0x05, 0xe6}
	colMenu        = color.NRGBA{0x05, 0x0a, 0x05, 0xff}
)

func (t *termuxTheme) Color(name fyne.ThemeColorName, _ fyne.ThemeVariant) color.Color {
	switch name {
	case theme.ColorNameBackground:
		return colBg
	case theme.ColorNameForeground:
		return colFg
	case theme.ColorNameButton:
		return colBtn
	case theme.ColorNameDisabledButton:
		return colBtnDisabled
	case theme.ColorNameDisabled:
		return colDisabled
	case theme.ColorNameError:
		return colRed
	case theme.ColorNameFocus:
		return colFocus
	case theme.ColorNameHeaderBackground:
		return colHeader
	case theme.ColorNameHover:
		return colHover
	case theme.ColorNameHyperlink:
		return colCyan
	case theme.ColorNameInputBackground:
		return colInput
	case theme.ColorNameInputBorder:
		return colInputBorder
	case theme.ColorNameMenuBackground:
		return colMenu
	case theme.ColorNameOverlayBackground:
		return colOverlay
	case theme.ColorNamePlaceHolder:
		return colFgDim
	case theme.ColorNamePressed:
		return colPressed
	case theme.ColorNamePrimary:
		return colFocus
	case theme.ColorNameScrollBar:
		return colScroll
	case theme.ColorNameSelection:
		return colSelection
	case theme.ColorNameSeparator:
		return colSeparator
	case theme.ColorNameShadow:
		return colShadow
	case theme.ColorNameSuccess:
		return colGreen
	case theme.ColorNameWarning:
		return colAmber
	}
	// Anything we missed — fall back to stock dark theme.
	return theme.DarkTheme().Color(name, theme.VariantDark)
}

// Font forces monospace everywhere for the terminal look.
func (t *termuxTheme) Font(style fyne.TextStyle) fyne.Resource {
	style.Monospace = true
	return theme.DarkTheme().Font(style)
}

func (t *termuxTheme) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DarkTheme().Icon(name)
}

// Size keeps padding tight for a dense terminal feel.
func (t *termuxTheme) Size(name fyne.ThemeSizeName) float32 {
	switch name {
	case theme.SizeNameText:
		return 13
	case theme.SizeNamePadding:
		return 3
	case theme.SizeNameInnerPadding:
		return 3
	}
	return theme.DarkTheme().Size(name)
}
