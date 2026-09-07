package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"

	demo "dma-is-back"
)

func run() error {
	ebiten.SetWindowSize(768, 540)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle("DMA is Back!")

	game := demo.NewGame()
	defer game.Cleanup()
	return ebiten.RunGame(game)
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}
