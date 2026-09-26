//go:build dck_fidelity_rendercheck

package dmaisback

import (
	"fmt"
	"os"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
	capture "github.com/olivierh59500/democonstructionkit/fidelity/ebiten"
)

// This opt-in native check captures the deterministic intro and main scene.
// Set DCK_DMA_CAPTURE_DIR to keep its PNGs for comparison with another revision.
func TestMain(m *testing.M) {
	if code := m.Run(); code != 0 {
		os.Exit(code)
	}
	directory := os.Getenv("DCK_DMA_CAPTURE_DIR")
	if directory == "" {
		var err error
		directory, err = os.MkdirTemp("", "dma-dck-capture-")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	var demo *Game
	err := capture.Run(capture.Config{
		Directory: directory,
		Frames:    []int{0, 1, 60, 240, 600, 1200, 2400},
		Width:     screenWidth,
		Height:    screenHeight,
	}, func() (ebiten.Game, error) {
		demo = NewGame()
		// Audio playback does not contribute to image motion in this production.
		demo.audioReady = true
		return demo, nil
	})
	if demo != nil {
		demo.Cleanup()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("DMA DCK captures: %s\n", directory)
}
