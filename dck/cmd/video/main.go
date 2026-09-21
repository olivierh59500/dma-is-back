// Command video exports the complete game canvas and its own audio.
package main

import (
	"flag"
	"log"
	"time"

	demo "dma-is-back/dck"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/olivierh59500/democonstructionkit/video"
)

func main() {
	config := video.Config{Output: "dma-is-back.mp4", Title: "DMA Is Back", Width: 768, Height: 540, FPS: 60, TPS: 60, SampleRate: 48000, Duration: 3 * time.Minute}
	config.Flags(flag.CommandLine)
	flag.Parse()
	if err := video.Run(config, func() (ebiten.Game, error) {
		return demo.NewGame(), nil
	}); err != nil {
		log.Fatal(err)
	}
}
