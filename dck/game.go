// Package dmaisback implements the DMA is Back demo for desktop and mobile.
package dmaisback

import (
	"bytes"
	originalassets "dma-is-back"
	"image"
	"image/color"
	_ "image/png"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	kit "github.com/olivierh59500/democonstructionkit"
	"github.com/olivierh59500/democonstructionkit/composite"
	"github.com/olivierh59500/democonstructionkit/effects"
	"github.com/olivierh59500/democonstructionkit/motion"
	"github.com/olivierh59500/democonstructionkit/presets"
	"github.com/olivierh59500/democonstructionkit/scrolling"
	"github.com/olivierh59500/democonstructionkit/sound"
	audio "github.com/olivierh59500/democonstructionkit/sound/output"
	"github.com/olivierh59500/democonstructionkit/timeline"
)

const (
	screenWidth     = 768
	screenHeight    = 540
	stCanvasWidth   = 640
	stCanvasHeight  = 400
	fadeSpeed       = .03
	posSpeed        = .014
	audioSampleRate = 48000
	introText       = "     " + "IF YOU THINK THIS IS ALL, YOU'RE SO WRONG..." + "     "
	mainText        = "     " + "     " + "WELCOME TO THE \"DMA-IS-BACK\" DEMO BY BILIZIR, WRITTEN IN GOLANG + EBITEN." + "     " +
		"GREETINGS TO ALL DEMOSCENE LOVERS AND ATARI ST FANS!" + "     " +
		"LET'S WRAP..." + "     " + "     "
)

var (
	backData  = originalassets.DCKAssetBackData()
	fontData  = originalassets.DCKAssetFontData()
	musicData = originalassets.DCKAssetMusicData()
)

// Game holds the production's assets, music and scene timing. Its text
// transports, image grid, CRT pass and animated cube belong to DCK.
type Game struct {
	cube                       *effects.JellyCube
	introScroll, mainScroll    *scrolling.Scrolling
	logoGrid                   composite.ImageGrid
	logoPath                   motion.NestedOrbit
	crt                        *effects.CRTOverlay
	backImg, fontImg, stCanvas *ebiten.Image
	pos                        float64
	scrollIteration            int
	handoff                    *timeline.IntroHandoff
	audioContext               *audio.Context
	audioPlayer                *audio.Player
	musicStream                *sound.Stream
	audioReady                 bool
	drawOp                     ebiten.DrawImageOptions
}

func NewGame() *Game {
	g := &Game{}
	var err error
	g.handoff, err = timeline.NewIntroHandoff(presets.FadedIntroHandoff(fadeSpeed, .1))
	if err != nil {
		panic(err)
	}
	g.cube, err = effects.NewJellyCube(effects.DMAJellyCubeConfig())
	if err != nil {
		panic(err)
	}
	g.loadImages()
	g.stCanvas = ebiten.NewImage(stCanvasWidth, stCanvasHeight)
	atlas, err := presets.FontAtlas("dma-is-back", g.fontImg)
	if err != nil {
		panic(err)
	}
	intro := presets.DMAIntroFeed(atlas, introText)
	g.introScroll, err = scrolling.New(scrolling.Config{Feed: &intro})
	if err != nil {
		panic(err)
	}
	main, err := presets.DMAScanlineScroll(atlas, mainText)
	if err != nil {
		panic(err)
	}
	g.mainScroll, err = scrolling.New(scrolling.Config{Scanline: &main})
	if err != nil {
		panic(err)
	}
	g.logoGrid = presets.DMALogoGrid()
	g.logoPath = motion.DefaultNestedOrbit(
		motion.Point{X: stCanvasWidth / 2, Y: stCanvasHeight / 2},
		motion.Point{X: stCanvasWidth / 4, Y: stCanvasHeight / 2.7},
	)
	g.crt, err = effects.NewCRTOverlay(presets.DMACRTOverlay())
	if err != nil {
		log.Printf("Failed to compile CRT shader: %v", err)
	}
	return g
}

// loadImages loads all image assets
func (g *Game) loadImages() {
	var err error

	// Load background/logo image
	img, _, err := image.Decode(bytes.NewReader(backData))
	if err != nil {
		log.Printf("Failed to load background image: %v", err)
	} else {
		g.backImg = ebiten.NewImageFromImage(img)
	}

	// Load font image
	img, _, err = image.Decode(bytes.NewReader(fontData))
	if err != nil {
		log.Printf("Failed to load font image: %v", err)
		// Create dummy font if loading fails
		g.fontImg = ebiten.NewImage(480, 216)
		g.fontImg.Fill(color.White)
	} else {
		g.fontImg = ebiten.NewImageFromImage(img)
	}
}

// initAudio opens the soundtrack and starts audio output.
func (g *Game) initAudio() {
	g.audioContext = audio.NewContext(audioSampleRate)

	// Let DCK choose and configure the music decoder.
	var err error
	g.musicStream, err = sound.Open("music.ym", musicData, sound.Options{SampleRate: audioSampleRate, Loop: true, PCMFormat: sound.Float32, Gain: 1})
	if err != nil {
		log.Printf("Failed to open music: %v", err)
		return
	}

	// Connect the shared stream to audio output.
	g.audioPlayer, err = g.audioContext.NewPlayerF32(g.musicStream)
	if err != nil {
		log.Printf("Failed to create audio player: %v", err)
		g.musicStream.Close()
		g.musicStream = nil
		return
	}

	// Set reasonable volume for YM music
	g.audioPlayer.SetVolume(0.7)
}

// Update advances the active transport and production timing once per tick.
func (g *Game) Update() error {
	if !g.audioReady {
		// Android installs its audio context after the first Ebitengine update.
		g.audioReady = true
		g.initAudio()
	}
	if !g.handoff.Main() {
		if err := g.introScroll.Update(kit.Frame{}); err != nil {
			return err
		}
		g.handoff.Step(g.introScroll.Finished())
		if g.handoff.JustEntered() {
			g.scrollIteration = 0
		}
		return nil
	}
	g.handoff.Step(false)
	if g.handoff.CueReady() && g.audioPlayer != nil {
		g.audioPlayer.Play()
		g.handoff.MarkCue()
	}
	g.scrollIteration++
	g.pos += posSpeed
	if err := g.mainScroll.Update(kit.Frame{Tick: uint64(g.scrollIteration)}); err != nil {
		return err
	}
	return g.cube.Update(kit.Frame{Time: float64(g.scrollIteration) / 60})
}

func (g *Game) releaseIntroResources() {
	if g.introScroll != nil {
		g.introScroll.Close()
		g.introScroll = nil
	}
	if g.crt != nil {
		g.crt.Close()
		g.crt = nil
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.Black)
	if !g.handoff.Main() {
		g.stCanvas.Fill(color.Black)
		y := float64(stCanvasHeight/2 - g.introScroll.Image().Bounds().Dy()/2)
		if g.crt != nil {
			g.crt.DrawAt(g.stCanvas, g.introScroll.Image(), 0, y)
		} else {
			g.drawOp.GeoM.Reset()
			g.drawOp.GeoM.Translate(0, y)
			g.stCanvas.DrawImage(g.introScroll.Image(), &g.drawOp)
		}
		g.drawOp.GeoM.Reset()
		g.drawOp.ColorScale.Reset()
		g.drawOp.GeoM.Translate(sceneOffsetX(screen.Bounds().Dx()), 70)
		screen.DrawImage(g.stCanvas, &g.drawOp)
		return
	}
	g.releaseIntroResources()
	g.mainScroll.Draw(g.stCanvas)
	if g.backImg != nil {
		point := g.logoPath.At(g.pos)
		g.logoGrid.DrawCentered(g.stCanvas, g.backImg, point.X, point.Y)
	}
	g.cube.Draw(g.stCanvas)
	g.drawOp.GeoM.Reset()
	g.drawOp.ColorScale.Reset()
	g.drawOp.GeoM.Translate(sceneOffsetX(screen.Bounds().Dx()), 70)
	g.drawOp.ColorScale.ScaleAlpha(float32(g.handoff.Fade()))
	screen.DrawImage(g.stCanvas, &g.drawOp)
}

func logicalWidth(outsideWidth, outsideHeight int) int {
	if outsideWidth <= 0 || outsideHeight <= 0 {
		return screenWidth
	}

	width := (outsideWidth*screenHeight + outsideHeight - 1) / outsideHeight
	if width < screenWidth {
		return screenWidth
	}
	const maxLogicalWidth = 1280
	if width > maxLogicalWidth {
		return maxLogicalWidth
	}
	return width
}

func sceneOffsetX(layoutWidth int) float64 {
	return float64(layoutWidth-stCanvasWidth) / 2
}

// Layout preserves the original aspect ratio and uses black side bands on
// wide displays instead of stretching the Atari ST canvas.
func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return logicalWidth(outsideWidth, outsideHeight), screenHeight
}

// Cleanup releases owned effects and the soundtrack when the game exits.
func (g *Game) Cleanup() {
	g.releaseIntroResources()
	if g.mainScroll != nil {
		g.mainScroll.Close()
	}
	if g.cube != nil {
		g.cube.Close()
	}
	if g.audioPlayer != nil {
		g.audioPlayer.Close()
	}
	if g.musicStream != nil {
		g.musicStream.Close()
	}
}

// SetSmoothTransitions selects continuous cube effects or the historical path.
func (g *Game) SetSmoothTransitions(enabled bool) { g.cube.SetSmoothTransitions(enabled) }
