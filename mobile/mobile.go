// Package mobile exposes the demo to ebitenmobile.
package mobile

import (
	demo "dma-is-back"

	enginemobile "github.com/hajimehoshi/ebiten/v2/mobile"
)

func init() {
	enginemobile.SetGame(demo.NewGame())
}

// Dummy forces gomobile to include this package in the Android binding.
func Dummy() {}
