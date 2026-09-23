// Package dmaisback implements the DMA is Back demo for desktop and mobile.
package dmaisback

import (
	"bytes"
	originalassets "dma-is-back"
	"image"
	"image/color"

	kit "github.com/olivierh59500/democonstructionkit"
	"github.com/olivierh59500/democonstructionkit/composite"
	"github.com/olivierh59500/democonstructionkit/effects"
	"github.com/olivierh59500/democonstructionkit/presets"
	"github.com/olivierh59500/democonstructionkit/scrolling"
	"github.com/olivierh59500/democonstructionkit/sound"

	_ "image/png"
	"log"
	"math"

	"github.com/hajimehoshi/ebiten/v2"

	audio "github.com/olivierh59500/democonstructionkit/sound/output"
)

const (
	// Screen dimensions
	screenWidth  = 768
	screenHeight = 540

	// Canvas dimensions
	stCanvasWidth  = 640
	stCanvasHeight = 400

	// Logo pattern dimensions
	// Two columns and four overlapping rows cover every animated viewport
	// position while keeping the backing texture small enough for mobile GPUs.
	logoPatternX = 2
	logoPatternY = 4
	logoTileW    = 640
	logoTileH    = 200

	// Animation parameters
	fadeSpeed       = 0.03 // Doubled from 0.01 for faster transitions
	scrollSpeed     = 8
	posSpeed        = 0.014
	audioSampleRate = 48000

	// Font parameters
	fontHeight     = 36
	introFontScale = 2.0 // Scale for intro text
	demoFontScale  = 3.0 // Scale for demo text
	scrollStripH   = int(demoFontScale)
	scrollStrips   = (stCanvasHeight + scrollStripH - 1) / scrollStripH
)

// Wave types for the distortion effects
const (
	cdZero = iota
	cdSlowSin
	cdMedSin
	cdFastSin
	cdSlowDist
	cdMedDist
	cdFastDist
	cdSplitted
)

// Canvas background colors.
var (
	blackFill      color.Color = color.Black
	mainScrollFill color.Color = color.RGBA{0x00, 0x00, 0x60, 0xFF}
)

// Embedded assets - resources compiled into the binary
var (
	backData = originalassets.DCKAssetBackData()

	fontData = originalassets.DCKAssetFontData()

	musicData = originalassets.

		// Letter represents a character in the bitmap font
		DCKAssetMusicData()
)

// CRT shader source - simulates old CRT monitor effects
const crtShaderSrc = `
package main

func Fragment(position vec4, texCoord vec2, color vec4) vec4 {
	var uv vec2
	uv = texCoord

	// Barrel distortion effect
	var dc vec2
	dc = uv - 0.5
	dc = dc * (1.0 + dot(dc, dc) * 0.15)
	uv = dc + 0.5

	// Check bounds
	if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
		return vec4(0.0, 0.0, 0.0, 1.0)
	}

	// Sample texture
	var col vec4
	col = imageSrc0At(uv)

	// Scanlines effect
	var scanline float
	scanline = sin(uv.y * 800.0) * 0.04
	col.rgb = col.rgb - scanline

	// RGB shift (chromatic aberration)
	var rShift float
	var bShift float
	rShift = imageSrc0At(uv + vec2(0.002, 0.0)).r
	bShift = imageSrc0At(uv - vec2(0.002, 0.0)).b
	col.r = rShift
	col.b = bShift

	// Vignette effect (darker edges)
	var vignette float
	vignette = 1.0 - dot(dc, dc) * 0.5
	col.rgb = col.rgb * vignette

	return col * color
}
`

// Game represents the main demo state
type Game struct {
	cube           *effects.JellyCube
	scrollRenderer *scrolling.Scrolling
	// Images
	backImg *ebiten.Image
	fontImg *ebiten.Image

	// Canvases for different rendering layers
	stCanvas       *ebiten.Image // Main ST screen canvas
	surfScroll     *ebiten.Image // Main demo scroll surface
	surfScroll1    *ebiten.Image // Current intro scroll surface
	surfScroll2    *ebiten.Image // Next intro scroll surface
	introViewport1 *ebiten.Image // Visible part of surfScroll1
	introViewport2 *ebiten.Image // Visible part of surfScroll2

	// Animation state
	fadeImg float64 // Fade alpha value
	pos     float64 // General position counter

	// Audio
	audioContext *audio.Context
	audioPlayer  *audio.Player
	musicStream  *sound.Stream
	audioReady   bool
	musicStarted bool

	// State flags
	introComplete bool

	// Shader
	crtShader *ebiten.Shader

	// Font data
	fontAtlas *scrolling.Atlas

	// Intro scrolling state
	introX      int
	introLetter int
	introTile   int
	introSpeed  int

	// Main scrolling state
	frontWavePos    int     // Current position in wave table
	letterNum       int     // Current letter index
	letterDecal     int     // Letter position offset
	curves          [][]int // Wave curve data
	frontMainWave   []int   // Combined wave data
	position        []int   // Text position data
	scrollTextRunes []rune  // Pre-converted runes for optimization
	introTextRunes  []rune  // Pre-converted runes for optimization
	scrollIteration int     // Frame counter
	waveByStrip     [scrollStrips]int
	scrollVertices  []ebiten.Vertex
	scrollIndices   []uint16

	// Optimization: reusable draw options
	drawOp        *ebiten.DrawImageOptions
	drawRectOp    *ebiten.DrawRectShaderOptions
	lastLetterNum int // Track last rendered letter number for caching

}

// NewGame creates and initializes a new game instance
func NewGame() *Game {
	g := &Game{
		fadeImg:       2.0,
		introX:        -1,
		introLetter:   -1,
		introTile:     -1,
		introSpeed:    scrollSpeed,
		drawOp:        &ebiten.DrawImageOptions{},
		drawRectOp:    &ebiten.DrawRectShaderOptions{},
		lastLetterNum: -1,
	}

	var err error
	g.cube, err = effects.NewJellyCube(effects.DMAJellyCubeConfig())
	if err != nil {
		panic(err)
	}
	g.scrollVertices = make([]ebiten.Vertex, 0, scrollStrips*8)
	g.scrollIndices = make([]uint16, 0, scrollStrips*12)

	// Initialize scrolling texts
	spc := "     "
	g.introTextRunes = []rune(spc +
		"IF YOU THINK THIS IS ALL, YOU'RE SO WRONG..." + spc)

	// Main demo text
	g.scrollTextRunes = []rune(spc + spc + "WELCOME TO THE \"DMA-IS-BACK\" DEMO BY BILIZIR, WRITTEN IN GOLANG + EBITEN." + spc +
		"GREETINGS TO ALL DEMOSCENE LOVERS AND ATARI ST FANS!" + spc +
		"LET'S WRAP..." + spc + spc)

	// Load images
	g.loadImages()

	// Create canvases
	g.stCanvas = ebiten.NewImage(stCanvasWidth, stCanvasHeight)
	g.surfScroll = ebiten.NewImage(int(float64(stCanvasWidth)*2.0), int(fontHeight*demoFontScale))
	g.surfScroll1 = ebiten.NewImage(stCanvasWidth+int(48*introFontScale), int(fontHeight*introFontScale))
	g.surfScroll2 = ebiten.NewImage(stCanvasWidth+int(48*introFontScale), int(fontHeight*introFontScale))
	viewport := image.Rect(0, 0, stCanvasWidth, int(fontHeight*introFontScale))
	g.introViewport1 = g.surfScroll1.SubImage(viewport).(*ebiten.Image)
	g.introViewport2 = g.surfScroll2.SubImage(viewport).(*ebiten.Image)

	// Load the shared proportional atlas recipe.
	g.fontAtlas, err = presets.FontAtlas("dma-is-back", g.fontImg)
	if err != nil {
		panic(err)
	}

	// Initialize wave curves for distortion effects
	g.curves = make([][]int, 8)
	g.createCurves()

	// Precalculate positions and waves
	g.precalcPosition()
	g.precalcMainWave()
	g.curves = nil

	// Compile CRT shader
	g.crtShader, err = ebiten.NewShader([]byte(crtShaderSrc))
	if err != nil {
		log.Printf("Failed to compile CRT shader: %v", err)
	}

	return g
}

// displayText renders text to scroll surface with scaling for demo
func (g *Game) displayText(letterOffset int) {
	if letterOffset == g.lastLetterNum {
		return
	}
	g.lastLetterNum = letterOffset
	g.surfScroll.Clear()
	if g.scrollRenderer == nil {
		glyphs := make([]scrolling.Glyph, len(g.scrollTextRunes))
		for i, r := range g.scrollTextRunes {
			if glyphImage, letter, ok := g.fontAtlas.ExactGlyph(r); ok {
				glyphs[i] = scrolling.Glyph{Image: glyphImage, Advance: float64(int(letter.Advance))}
			}
		}
		var err error
		g.scrollRenderer, err = scrolling.New(scrolling.Config{Glyphs: glyphs})
		if err != nil {
			panic(err)
		}
	}
	state := g.scrollRenderer.Window(letterOffset, float64(g.surfScroll.Bounds().Dx())/demoFontScale)
	state.ScaleX = demoFontScale
	state.ScaleY = demoFontScale
	state.X *= demoFontScale
	state.Options = *g.drawOp
	state.Options.GeoM.Reset()
	g.scrollRenderer.DrawAt(g.surfScroll, state)
}

// createCurves generates the wave curves for distortion effects
func (g *Game) createCurves() {
	curves, err := presets.RibbonCurves(1)
	if err != nil {
		panic(err)
	}
	g.curves = curves[:8]
}

func (g *Game) precalcPosition() {
	count := 0
	g.position = make([]int, 0, len(g.scrollTextRunes))

	for _, r := range g.scrollTextRunes {
		if _, letter, ok := g.fontAtlas.ExactGlyph(r); ok {
			count += int(float64(int(letter.Advance)) * demoFontScale)
			g.position = append(g.position, count)
		}
	}
}

// precalcMainWave precalculates wave data for main scroll
func (g *Game) precalcMainWave() {
	// Wave sequence for main demo
	frontMainWaveTable := []int{
		cdSlowSin, cdSlowSin, cdSlowDist, cdSlowSin,
		cdSlowSin, cdMedSin, cdFastSin, cdMedSin,
		cdSlowSin, cdMedDist, cdMedSin, cdSlowSin,
		cdSplitted,
	}

	var err error
	g.frontMainWave, err = composite.JoinDeltaCurves(g.curves, frontMainWaveTable)
	if err != nil {
		panic(err)
	}
}

func (g *Game) getSum(arr []int, index, decal int) int {
	return composite.CumulativeAt(arr, index, decal)
}

func (g *Game) updateWaveByStrip() {
	composite.FillCumulative(g.waveByStrip[:], g.frontMainWave, g.frontWavePos, 0)
}

func (g *Game) getPosition(i int) int {
	if i > 0 && i <= len(g.position) {
		return g.getSum(g.position, i-1, 0)
	}
	return 0
}

// getLetter gets letter at position with wrapping (optimized)
func (g *Game) getLetter(pos int) rune {
	if len(g.scrollTextRunes) == 0 {
		return ' '
	}
	return g.scrollTextRunes[pos%len(g.scrollTextRunes)]
}

// getIntroLetter gets intro letter at position with wrapping (optimized)
func (g *Game) getIntroLetter(pos int) rune {
	if len(g.introTextRunes) == 0 {
		return ' '
	}
	return g.introTextRunes[pos%len(g.introTextRunes)]
}

// animIntro handles intro animation
func (g *Game) animIntro() {
	// Check if we need to advance to next letter
	if g.introX < 0 {
		if g.introTile > -1 {
			char := g.getIntroLetter(g.introTile)
			if _, letter, ok := g.fontAtlas.ExactGlyph(char); ok {
				g.introX += int(float64(int(letter.Advance)) * introFontScale)
			}
		}
		g.introLetter++
		if g.introLetter >= len(g.introTextRunes) {
			g.introComplete = true
			g.fadeImg = 0
			g.scrollIteration = 0
			return
		}
		g.introTile = g.introLetter
	}
	g.introX -= g.introSpeed

	// Build the next scrolling surface, then swap the two buffers. Copying the
	// result back into surfScroll1 would add a clear and a full-surface draw on
	// every intro frame.
	g.surfScroll2.Clear()
	srcRect := image.Rect(g.introSpeed, 0, g.surfScroll1.Bounds().Dx(), int(fontHeight*introFontScale))
	g.drawOp.GeoM.Reset()
	g.drawOp.ColorScale.Reset()
	g.surfScroll2.DrawImage(g.surfScroll1.SubImage(srcRect).(*ebiten.Image), g.drawOp)

	// Draw new letter
	char := g.getIntroLetter(g.introTile)
	if glyphImage, _, ok := g.fontAtlas.ExactGlyph(char); ok {
		g.drawOp.GeoM.Reset()
		g.drawOp.GeoM.Scale(introFontScale, introFontScale)
		g.drawOp.GeoM.Translate(float64(stCanvasWidth+g.introX), 0)
		g.surfScroll2.DrawImage(glyphImage, g.drawOp)
	}

	g.surfScroll1, g.surfScroll2 = g.surfScroll2, g.surfScroll1
	g.introViewport1, g.introViewport2 = g.introViewport2, g.introViewport1
}

// drawIntroWithShader draws the intro scroll with CRT shader effect
func (g *Game) drawIntroWithShader() {
	g.stCanvas.Fill(blackFill)

	if g.crtShader != nil {
		g.drawRectOp.Images[0] = g.introViewport1
		g.drawRectOp.GeoM.Reset()
		g.drawRectOp.GeoM.Translate(0, float64(stCanvasHeight/2-int(fontHeight*introFontScale)/2))

		g.stCanvas.DrawRectShader(stCanvasWidth, int(fontHeight*introFontScale), g.crtShader, g.drawRectOp)
	} else {
		// Fallback without shader
		g.drawOp.GeoM.Reset()
		g.drawOp.GeoM.Translate(0, float64(stCanvasHeight/2-int(fontHeight*introFontScale)/2))
		g.stCanvas.DrawImage(g.introViewport1, g.drawOp)
	}
}

// drawMainScroll draws the main demo scrolling text with distortion
func (g *Game) drawMainScroll() {
	// Update wave position
	g.frontWavePos = g.scrollIteration * 10
	g.updateWaveByStrip()

	// Calculate horizontal offset
	decalX := g.waveByStrip[0]
	for strip := 1; strip < fontHeight; strip++ {
		c := g.waveByStrip[strip]
		if c < decalX {
			decalX = c
		}
	}

	if decalX < 0 {
		decalX = 0
	}

	// Calculate first visible letter
	i := 0
	dir := 0
	if decalX > g.letterDecal {
		dir = 1
	} else if decalX < g.letterDecal {
		dir = -1
	}

	for decalX < g.getPosition(g.letterNum+i) || g.getPosition(g.letterNum+i+1) <= decalX {
		i += dir
		if g.letterNum+i < 0 || g.letterNum+i >= len(g.position) {
			break
		}
	}
	g.letterNum += i
	if g.letterNum < 0 {
		g.letterNum = 0
	} else if g.letterNum >= len(g.position) {
		g.letterNum = len(g.position) - 1
	}
	g.letterDecal = g.getPosition(g.letterNum)

	// Render text to scroll surface
	g.displayText(g.letterNum)

	// Calculate bounce effect
	bounce := int(18.0 * math.Abs(math.Sin(float64(g.scrollIteration)*0.1)))

	// Get scroll surface dimensions
	scrollWidth := g.surfScroll.Bounds().Dx()

	// Clear canvas with blue background
	g.stCanvas.Fill(mainScrollFill)

	// Three adjacent output lines always use adjacent rows from the source.
	// Emit one quad for the whole strip instead of three one-pixel sub-images.
	g.scrollVertices = g.scrollVertices[:0]
	g.scrollIndices = g.scrollIndices[:0]
	for strip := 0; strip < scrollStrips; strip++ {
		dstY := strip * scrollStripH
		height := min(scrollStripH, stCanvasHeight-dstY)
		// Calculate wave-based horizontal offset (do not wrap negatives)
		frontWave := g.waveByStrip[strip]
		scrollXRaw := frontWave - g.letterDecal

		// Calculate source line with bounce effect
		srcY := ((strip + bounce) % fontHeight) * scrollStripH

		// Drawing rules:
		// - If scrollXRaw < 0: clamp (no wrap). Leave left side empty and draw the visible right part.
		// - Else: allow wrapping as before.
		if scrollXRaw < 0 {
			// Visible width after clamping (part of the text entering from the right)
			visibleWidth := stCanvasWidth + scrollXRaw // scrollXRaw is negative here
			if visibleWidth > 0 {
				g.appendScrollQuad(-scrollXRaw, dstY, 0, srcY, visibleWidth, height)
			}
			continue
		}

		// Non-negative offset: apply wrapping logic
		scrollX := scrollXRaw % scrollWidth
		if scrollX >= scrollWidth-stCanvasWidth {
			// Near end, need to wrap
			width1 := scrollWidth - scrollX
			g.appendScrollQuad(0, dstY, scrollX, srcY, width1, height)

			// Draw beginning to fill the rest
			width2 := stCanvasWidth - width1
			if width2 > 0 {
				g.appendScrollQuad(width1, dstY, 0, srcY, width2, height)
			}
		} else {
			// Normal case - draw full width
			g.appendScrollQuad(0, dstY, scrollX, srcY, stCanvasWidth, height)
		}
	}

	if len(g.scrollIndices) > 0 {
		g.stCanvas.DrawTriangles(g.scrollVertices, g.scrollIndices, g.surfScroll, nil)
	}
}

func (g *Game) appendScrollQuad(dstX, dstY, srcX, srcY, width, height int) {
	base := uint16(len(g.scrollVertices))
	x0, y0 := float32(dstX), float32(dstY)
	x1, y1 := float32(dstX+width), float32(dstY+height)
	sx0, sy0 := float32(srcX), float32(srcY)
	sx1, sy1 := float32(srcX+width), float32(srcY+height)

	g.scrollVertices = append(g.scrollVertices,
		ebiten.Vertex{DstX: x0, DstY: y0, SrcX: sx0, SrcY: sy0, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
		ebiten.Vertex{DstX: x1, DstY: y0, SrcX: sx1, SrcY: sy0, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
		ebiten.Vertex{DstX: x1, DstY: y1, SrcX: sx1, SrcY: sy1, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
		ebiten.Vertex{DstX: x0, DstY: y1, SrcX: sx0, SrcY: sy1, ColorR: 1, ColorG: 1, ColorB: 1, ColorA: 1},
	)
	g.scrollIndices = append(g.scrollIndices,
		base, base+1, base+2,
		base, base+2, base+3,
	)
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

// Update updates the game state
func (g *Game) Update() error {
	if !g.audioReady {
		// The Android activity installs the gomobile context and Ebiten view
		// before the first update. Opening the audio device here avoids doing
		// so too early while libgojni is still being loaded.
		g.audioReady = true
		g.initAudio()
	}

	if !g.introComplete {
		// Update intro animation
		g.animIntro()
	} else {
		// Main demo update

		// Fade in main scene
		if g.fadeImg < 1 {
			g.fadeImg += fadeSpeed
			if g.fadeImg > 1 {
				g.fadeImg = 1
			}
		}

		// Start music when demo begins
		if g.fadeImg > 0.1 && g.audioPlayer != nil && !g.musicStarted {
			g.audioPlayer.Play()
			g.musicStarted = true
		}

		// Update scroll position
		g.scrollIteration++

		// Update background animation
		g.pos += posSpeed

		if err := g.cube.Update(kit.Frame{Time: float64(g.scrollIteration) / 60}); err != nil {
			return err
		}
	}

	return nil
}

func (g *Game) releaseIntroResources() {
	if g.surfScroll1 == nil {
		return
	}

	g.drawRectOp.Images[0] = nil
	g.introViewport1 = nil
	g.introViewport2 = nil
	g.surfScroll1.Deallocate()
	g.surfScroll2.Deallocate()
	g.surfScroll1 = nil
	g.surfScroll2 = nil
	g.introTextRunes = nil
	if g.crtShader != nil {
		g.crtShader.Deallocate()
		g.crtShader = nil
	}
}

func (g *Game) drawAnimatedLogo() {
	if g.backImg == nil {
		return
	}

	x := stCanvasWidth/2 + (stCanvasWidth/4)*math.Cos(g.pos*4-math.Cos(g.pos-0.1))
	y := stCanvasHeight/2 + (stCanvasHeight/2.7)*-math.Sin(g.pos*2.3-math.Cos(g.pos-0.1))
	originX := x - float64(logoPatternX*logoTileW)/2
	originY := y - float64(logoPatternY*logoTileH)/2

	g.drawOp.ColorScale.Reset()
	for tileY := 0; tileY < logoPatternY; tileY++ {
		for tileX := 0; tileX < logoPatternX; tileX++ {
			g.drawOp.GeoM.Reset()
			g.drawOp.GeoM.Translate(
				originX+float64(tileX*logoTileW),
				originY+float64(tileY*logoTileH),
			)
			g.stCanvas.DrawImage(g.backImg, g.drawOp)
		}
	}
}

// Draw renders the game
func (g *Game) Draw(screen *ebiten.Image) {
	if !g.introComplete {
		// Draw intro phase
		screen.Fill(blackFill)

		// Draw intro scroll with CRT effect
		g.drawIntroWithShader()

		// Draw the intro canvas
		g.drawOp.GeoM.Reset()
		g.drawOp.ColorScale.Reset()
		g.drawOp.GeoM.Translate(sceneOffsetX(screen.Bounds().Dx()), 70)
		screen.DrawImage(g.stCanvas, g.drawOp)
	} else {
		g.releaseIntroResources()

		// Draw main demo
		screen.Fill(blackFill)

		// Draw distorted scrolling text (background layer)
		g.drawMainScroll()

		// Draw animated background logo
		g.drawAnimatedLogo()

		// The shared effect owns entrance timing, all modes and continuous handoffs.
		g.cube.Draw(g.stCanvas)

		// Final composite with fade
		g.drawOp.GeoM.Reset()
		g.drawOp.ColorScale.Reset()
		g.drawOp.GeoM.Translate(sceneOffsetX(screen.Bounds().Dx()), 70)
		g.drawOp.ColorScale.ScaleAlpha(float32(g.fadeImg))
		screen.DrawImage(g.stCanvas, g.drawOp)
	}
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

// Cleanup releases resources when game exits
func (g *Game) Cleanup() {
	if g.cube != nil {
		g.cube.Close()
	}
	if g.audioPlayer != nil {
		g.audioPlayer.Close()
	}
	if g.musicStream != nil {
		g.musicStream.Close()
	}
	if g.crtShader != nil {
		g.crtShader.Deallocate()
	}
}

// SetSmoothTransitions selects continuous cube effects or the historical path.
// Configure this before the first Update; the DCK version enables it by default.
func (g *Game) SetSmoothTransitions(enabled bool) { g.cube.SetSmoothTransitions(enabled) }
