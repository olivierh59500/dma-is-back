// Package dmaisback implements the DMA is Back demo for desktop and mobile.
package dmaisback

import originalassets "dma-is-back"

import (
	"bytes"
	"github.com/olivierh59500/democonstructionkit/geometry"

	"fmt"
	"github.com/olivierh59500/democonstructionkit/scrolling"
	"image"
	"image/color"
	_ "image/png"
	"io"
	"log"
	"math"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/audio"
	"github.com/olivierh59500/ym-player/pkg/stsound"
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
	rotationSpeed   = 0.05
	zoomSpeed       = 0.01
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

const (
	// Cube rotation modes
	rotationModeNormal = iota
	rotationModeTumble
	rotationModePulsate
	rotationModeSwing
	rotationModeBounce
	rotationModeTotal // Total number of modes
)

// Color definitions for 3D cube faces
var (
	col0 = color.RGBA{0xE0, 0xA0, 0xC0, 0xFF}
	col1 = color.RGBA{0xE0, 0x60, 0xC0, 0xFF}
	col2 = color.RGBA{0xE0, 0xE0, 0xE0, 0xFF}

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

type Letter struct {
	x, y  int
	width int
	image *ebiten.Image
}

// Vector3 represents a 3D point in space
type Vector3 = geometry.Vec3

// Face represents a quad face with 4 vertices and a color
type Face struct {
	P1, P2, P3, P4 int
	Color          color.RGBA
}

// faceWithDepth is used for depth sorting
type faceWithDepth struct {
	face  Face
	depth float64
}

// YMPlayer wraps the YM player for use with Ebiten's audio system
type YMPlayer struct {
	player       *stsound.StSound
	buffer       []int16
	mutex        sync.Mutex
	position     int64
	totalSamples int64
	loop         bool
}

type ymFloat32Reader struct {
	player *YMPlayer
}

func (r *ymFloat32Reader) Read(p []byte) (int, error) {
	return r.player.readFloat32(p)
}

// NewYMPlayer creates a new YM player instance
func NewYMPlayer(data []byte, sampleRate int, loop bool) (*YMPlayer, error) {
	// Create YM player with specified sample rate
	player := stsound.CreateWithRate(sampleRate)

	// Load YM data from memory
	if err := player.LoadMemory(data); err != nil {
		player.Destroy()
		return nil, fmt.Errorf("failed to load YM data: %w", err)
	}

	// Enable looping if requested
	player.SetLoopMode(loop)

	// Get music info for duration calculation
	info := player.GetInfo()
	totalSamples := int64(info.MusicTimeInMs) * int64(sampleRate) / 1000

	return &YMPlayer{
		player:       player,
		buffer:       make([]int16, 4096), // Audio buffer size
		totalSamples: totalSamples,
		loop:         loop,
	}, nil
}

// Read implements io.Reader for audio streaming
func (y *YMPlayer) Read(p []byte) (n int, err error) {
	y.mutex.Lock()
	defer y.mutex.Unlock()
	if y.player == nil {
		return 0, io.EOF
	}

	// Calculate how many samples we need (2 bytes per sample, stereo)
	samplesNeeded := len(p) / 4
	if samplesNeeded == 0 {
		return 0, nil
	}

	// Process audio in chunks
	processed := 0
	for processed < samplesNeeded {
		// Calculate chunk size
		chunkSize := samplesNeeded - processed
		if chunkSize > len(y.buffer) {
			chunkSize = len(y.buffer)
		}

		// Generate audio samples from YM player
		if !y.player.Compute(y.buffer[:chunkSize], chunkSize) {
			if !y.loop {
				// End of music, fill with silence
				clear(p[processed*4 : samplesNeeded*4])
				err = io.EOF
				break
			}
			// Loop enabled, music will restart automatically
		}

		// Convert mono to stereo. Volume is handled by audio.Player.
		for i := 0; i < chunkSize; i++ {
			sample := y.buffer[i]
			offset := (processed + i) * 4
			p[offset] = byte(sample)
			p[offset+1] = byte(sample >> 8)
			p[offset+2] = byte(sample)
			p[offset+3] = byte(sample >> 8)
		}

		processed += chunkSize
		y.position += int64(chunkSize)
	}

	return samplesNeeded * 4, err
}

// readFloat32 streams the YM output in Ebitengine's native float32 format.
// The mono sample is converted once and then copied to both stereo channels.
func (y *YMPlayer) readFloat32(p []byte) (n int, err error) {
	y.mutex.Lock()
	defer y.mutex.Unlock()
	if y.player == nil {
		return 0, io.EOF
	}

	framesNeeded := len(p) / 8
	if framesNeeded == 0 {
		return 0, nil
	}

	processed := 0
	for processed < framesNeeded {
		chunkSize := min(framesNeeded-processed, len(y.buffer))
		if !y.player.Compute(y.buffer[:chunkSize], chunkSize) && !y.loop {
			clear(p[processed*8 : framesNeeded*8])
			err = io.EOF
			break
		}

		for i := 0; i < chunkSize; i++ {
			sample := math.Float32bits(float32(y.buffer[i]) / (1 << 15))
			offset := (processed + i) * 8
			p[offset] = byte(sample)
			p[offset+1] = byte(sample >> 8)
			p[offset+2] = byte(sample >> 16)
			p[offset+3] = byte(sample >> 24)
			p[offset+4] = byte(sample)
			p[offset+5] = byte(sample >> 8)
			p[offset+6] = byte(sample >> 16)
			p[offset+7] = byte(sample >> 24)
		}

		processed += chunkSize
		y.position += int64(chunkSize)
	}

	return framesNeeded * 8, err
}

// Seek implements io.Seeker for positioning in the audio stream
func (y *YMPlayer) Seek(offset int64, whence int) (int64, error) {
	y.mutex.Lock()
	defer y.mutex.Unlock()

	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = y.position + offset
	case io.SeekEnd:
		newPos = y.totalSamples + offset
	default:
		return 0, fmt.Errorf("invalid whence: %d", whence)
	}

	// Clamp position to valid range
	if newPos < 0 {
		newPos = 0
	}
	if newPos > y.totalSamples {
		newPos = y.totalSamples
	}

	// Note: The ym-player library doesn't support seeking, so we just update our position
	// The player will continue from where it is, which is fine for our use case
	// since we're using it with infinite loop enabled
	y.position = newPos

	return newPos, nil
}

// Close releases resources used by the YM player
func (y *YMPlayer) Close() error {
	y.mutex.Lock()
	defer y.mutex.Unlock()

	if y.player != nil {
		y.player.Destroy()
		y.player = nil
	}
	return nil
}

// Length returns the total length of the music in samples
func (y *YMPlayer) Length() int64 {
	return y.totalSamples
}

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
	smoothCubeTransitions      bool
	cubeHandoff                geometry.Handoff
	cubePrevious, cubeIncoming []Vector3
	cubeTick                   int
	cubeChanged                bool
	swingOriginX, swingOriginZ float64
	scrollRenderer             *scrolling.Scrolling
	// Images
	backImg  *ebiten.Image
	fontImg  *ebiten.Image
	whiteImg *ebiten.Image

	// Canvases for different rendering layers
	stCanvas       *ebiten.Image // Main ST screen canvas
	surfScroll     *ebiten.Image // Main demo scroll surface
	surfScroll1    *ebiten.Image // Current intro scroll surface
	surfScroll2    *ebiten.Image // Next intro scroll surface
	introViewport1 *ebiten.Image // Visible part of surfScroll1
	introViewport2 *ebiten.Image // Visible part of surfScroll2

	// Animation state
	fadeImg  float64 // Fade alpha value
	pos      float64 // General position counter
	zoom3d   float64 // 3D cube zoom factor
	rotation Vector3 // 3D cube rotation angles

	// 3D cube data
	vertices            []Vector3
	faces               []Face
	transformedVertices []Vector3       // Pre-allocated for optimization
	facesWithDepth      []faceWithDepth // Pre-allocated for optimization
	cubeVertices        []ebiten.Vertex
	cubeIndices         []uint16

	// Audio
	audioContext *audio.Context
	audioPlayer  *audio.Player
	ymPlayer     *YMPlayer
	audioReady   bool
	musicStarted bool

	// State flags
	introComplete bool

	// Shader
	crtShader *ebiten.Shader

	// Font data
	letterData map[rune]*Letter

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

	// New fields for cube movements
	rotationMode     int     // Current rotation mode
	rotationTimer    float64 // Timer to change mode
	rotationDuration float64 // Duration of each mode
	pulsePhase       float64 // Phase for pulsation effect
	swingAmplitude   float64 // Swing amplitude
	bounceVelocity   float64 // Bounce velocity
	bouncePosition   float64 // Bounce position
}

// NewGame creates and initializes a new game instance
func NewGame() *Game {
	g := &Game{
		smoothCubeTransitions: true,
		fadeImg:               2.0,
		zoom3d:                0.0,
		letterData:            make(map[rune]*Letter),
		introX:                -1,
		introLetter:           -1,
		introTile:             -1,
		introSpeed:            scrollSpeed,
		drawOp:                &ebiten.DrawImageOptions{},
		drawRectOp:            &ebiten.DrawRectShaderOptions{},
		lastLetterNum:         -1,

		// New fields
		rotationMode:     rotationModeNormal,
		rotationTimer:    0,
		rotationDuration: 300, // Change mode every 300 frames (~5 seconds)
		pulsePhase:       0,
		swingAmplitude:   1.0,
		bounceVelocity:   0,
		bouncePosition:   0,
	}

	// Initialize 3D cube vertices - perfect cube with equal dimensions
	size := 80.0
	g.vertices = []Vector3{
		{-size, -size, -size}, // 0 - back bottom left
		{size, -size, -size},  // 1 - back bottom right
		{size, size, -size},   // 2 - back top right
		{-size, size, -size},  // 3 - back top left
		{-size, -size, size},  // 4 - front bottom left
		{size, -size, size},   // 5 - front bottom right
		{size, size, size},    // 6 - front top right
		{-size, size, size},   // 7 - front top left
	}

	// Initialize cube faces with proper winding order
	g.faces = []Face{
		{4, 5, 6, 7, col0}, // Front face
		{1, 0, 3, 2, col0}, // Back face
		{5, 1, 2, 6, col1}, // Right face
		{0, 4, 7, 3, col1}, // Left face
		{7, 6, 2, 3, col2}, // Top face
		{0, 1, 5, 4, col2}, // Bottom face
	}

	// Pre-allocate transformation buffers
	g.transformedVertices = make([]Vector3, len(g.vertices))
	g.facesWithDepth = make([]faceWithDepth, len(g.faces))
	g.cubeVertices = make([]ebiten.Vertex, 0, len(g.faces)*4)
	g.cubeIndices = make([]uint16, 0, len(g.faces)*6)
	g.scrollVertices = make([]ebiten.Vertex, 0, scrollStrips*8)
	g.scrollIndices = make([]uint16, 0, scrollStrips*12)
	g.whiteImg = ebiten.NewImage(1, 1)
	g.whiteImg.Fill(color.White)

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

	// Initialize font data
	g.initFontData()

	// Initialize wave curves for distortion effects
	g.curves = make([][]int, 8)
	g.createCurves()

	// Precalculate positions and waves
	g.precalcPosition()
	g.precalcMainWave()
	g.curves = nil

	// Compile CRT shader
	var err error
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
			if letter, ok := g.letterData[r]; ok {
				glyphs[i] = scrolling.Glyph{Image: letter.image, Advance: float64(letter.width)}
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

// initFontData initializes the bitmap font character data
func (g *Game) initFontData() {
	// Font character definitions - position and width in the font bitmap
	data := []struct {
		char  rune
		x, y  int
		width int
	}{
		{' ', 0, 0, 32},
		{'!', 48, 0, 16},
		{'"', 96, 0, 32},
		{'\'', 336, 0, 16},
		{'(', 384, 0, 32},
		{')', 432, 0, 32},
		{'+', 48, 36, 48},
		{',', 96, 36, 16},
		{'-', 144, 36, 32},
		{'.', 192, 36, 16},
		{'0', 288, 36, 48},
		{'1', 336, 36, 48},
		{'2', 384, 36, 48},
		{'3', 432, 36, 48},
		{'4', 0, 72, 48},
		{'5', 48, 72, 48},
		{'6', 96, 72, 48},
		{'7', 144, 72, 48},
		{'8', 192, 72, 48},
		{'9', 240, 72, 48},
		{':', 288, 72, 16},
		{';', 336, 72, 16},
		{'<', 384, 72, 32},
		{'=', 432, 72, 32},
		{'>', 0, 108, 32},
		{'?', 48, 108, 48},
		{'A', 144, 108, 48},
		{'B', 192, 108, 48},
		{'C', 240, 108, 48},
		{'D', 288, 108, 48},
		{'E', 336, 108, 48},
		{'F', 384, 108, 48},
		{'G', 432, 108, 48},
		{'H', 0, 144, 48},
		{'I', 48, 144, 16},
		{'J', 96, 144, 48},
		{'K', 144, 144, 48},
		{'L', 192, 144, 48},
		{'M', 240, 144, 48},
		{'N', 288, 144, 48},
		{'O', 336, 144, 48},
		{'P', 384, 144, 48},
		{'Q', 432, 144, 48},
		{'R', 0, 180, 48},
		{'S', 48, 180, 48},
		{'T', 96, 180, 48},
		{'U', 144, 180, 48},
		{'V', 192, 180, 48},
		{'W', 240, 180, 48},
		{'X', 288, 180, 48},
		{'Y', 336, 180, 48},
		{'Z', 384, 180, 48},
	}

	// Build character lookup map
	for _, d := range data {
		g.letterData[d.char] = &Letter{
			x:     d.x,
			y:     d.y,
			width: d.width,
			image: g.fontImg.SubImage(image.Rect(d.x, d.y, d.x+d.width, d.y+fontHeight)).(*ebiten.Image),
		}
	}
}

// createCurves generates the wave curves for distortion effects
func (g *Game) createCurves() {
	for funcType := 0; funcType <= 7; funcType++ {
		var step, progress float64

		// Set parameters for each wave type
		switch funcType {
		case cdZero:
			step, progress = 2.25, 0
		case cdSlowSin:
			step, progress = 0.20, 140
		case cdMedSin:
			step, progress = 0.25, 175
		case cdFastSin:
			step, progress = 0.30, 210
		case cdSlowDist:
			step, progress = 0.12, 175
		case cdMedDist:
			step, progress = 0.16, 210
		case cdFastDist:
			step, progress = 0.20, 245
		case cdSplitted:
			step, progress = 0.18, 0
		}
		maxAngle := 360.0
		if funcType == cdSplitted {
			maxAngle = 720.0
		}
		local := make([]float64, 0, int(math.Ceil(maxAngle/step)))
		decal := 0.0
		previous := 0

		// Generate wave values
		for i := 0.0; i < maxAngle-step; i += step {
			val := 0.0
			rad := i * math.Pi / 180

			// Calculate wave value based on type
			switch funcType {
			case cdZero:
				val = 0
			case cdSlowSin:
				val = 100 * math.Sin(rad)
			case cdMedSin:
				val = 110 * math.Sin(rad)
			case cdFastSin:
				val = 120 * math.Sin(rad)
			case cdSlowDist:
				val = 100*math.Sin(rad) + 25.0*math.Sin(rad*10)
			case cdMedDist:
				val = 110*math.Sin(rad) + 27.5*math.Sin(rad*9)
			case cdFastDist:
				val = 120*math.Sin(rad) + 30.0*math.Sin(rad*8)
			case cdSplitted:
				dir := 1.0
				if len(local)%2 == 1 {
					dir = -1.0
				}
				amp := 12.0
				if i < 160 {
					amp *= i / 160
				} else if (720 - 160) < i {
					amp *= (720 - i) / 160
				}
				val = 90*math.Sin(rad) + dir*amp*math.Sin(rad*3)
			}
			local = append(local, val)
		}

		// Convert to delta values
		g.curves[funcType] = make([]int, len(local))
		for i := 0; i < len(local); i++ {
			nitem := -int(math.Floor(local[i] - decal))
			g.curves[funcType][i] = nitem - previous
			previous = nitem
			decal += progress / float64(len(local))
		}
	}
}

// precalcPosition precalculates text positions with scaled font
func (g *Game) precalcPosition() {
	count := 0
	g.position = make([]int, 0, len(g.scrollTextRunes))

	for _, r := range g.scrollTextRunes {
		if letter, ok := g.letterData[r]; ok {
			count += int(float64(letter.width) * demoFontScale)
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

	count := 0
	totalLength := 0
	for _, waveType := range frontMainWaveTable {
		totalLength += len(g.curves[waveType])
	}
	g.frontMainWave = make([]int, 0, totalLength)

	// Build combined wave from sequence
	for _, waveType := range frontMainWaveTable {
		wave := g.curves[waveType]
		for _, val := range wave {
			count += val
			g.frontMainWave = append(g.frontMainWave, count)
		}
	}
}

// getSum calculates sum with wrapping
func (g *Game) getSum(arr []int, index, decal int) int {
	n := len(arr)
	if n == 0 {
		return decal
	}

	maxVal := arr[n-1]
	f := index / n
	m := index % n
	return decal + f*maxVal + arr[m]
}

// updateWaveByStrip computes the horizontal displacement once per three-pixel
// strip. The previous scanline loop recalculated the same value three times
// and performed a division and modulo for every lookup.
func (g *Game) updateWaveByStrip() {
	wave := g.frontMainWave
	if len(wave) == 0 {
		clear(g.waveByStrip[:])
		return
	}

	index := g.frontWavePos
	cycle := index / len(wave)
	position := index % len(wave)
	offset := cycle * wave[len(wave)-1]
	for i := range g.waveByStrip {
		g.waveByStrip[i] = offset + wave[position]
		position++
		if position == len(wave) {
			position = 0
			offset += wave[len(wave)-1]
		}
	}
}

// getPosition gets text position
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
			if letter, ok := g.letterData[char]; ok {
				g.introX += int(float64(letter.width) * introFontScale)
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
	if letter, ok := g.letterData[char]; ok {
		g.drawOp.GeoM.Reset()
		g.drawOp.GeoM.Scale(introFontScale, introFontScale)
		g.drawOp.GeoM.Translate(float64(stCanvasWidth+g.introX), 0)
		g.surfScroll2.DrawImage(letter.image, g.drawOp)
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

// initAudio initializes the audio system with YM music
func (g *Game) initAudio() {
	g.audioContext = audio.NewContext(audioSampleRate)

	// Create YM player
	var err error
	g.ymPlayer, err = NewYMPlayer(musicData, audioSampleRate, true)
	if err != nil {
		log.Printf("Failed to create YM player: %v", err)
		return
	}

	// Create audio player from YM player
	g.audioPlayer, err = g.audioContext.NewPlayerF32(&ymFloat32Reader{player: g.ymPlayer})
	if err != nil {
		log.Printf("Failed to create audio player: %v", err)
		g.ymPlayer.Close()
		g.ymPlayer = nil
		return
	}

	// Set reasonable volume for YM music
	g.audioPlayer.SetVolume(0.7)
}

// Update updates the game state
func (g *Game) Update() error {
	if len(g.cubePrevious) != len(g.vertices) {
		g.cubePrevious = make([]Vector3, len(g.vertices))
		g.cubeIncoming = make([]Vector3, len(g.vertices))
	}
	g.sampleCube(g.cubePrevious)
	g.cubeChanged = false
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

		// Update 3D cube after delay
		if g.scrollIteration > 25 {
			// Handle rotation mode changes
			g.rotationTimer++
			if g.rotationTimer >= g.rotationDuration {
				g.rotationTimer = 0
				// Smooth transition between modes
				g.rotationMode = (g.rotationMode + 1) % rotationModeTotal
				g.cubeChanged = true

				// Reset parameters based on new mode
				switch g.rotationMode {
				case rotationModeBounce:
					g.bounceVelocity = 0.08
					g.bouncePosition = 0
				case rotationModeSwing:
					// Adjust initial rotation to avoid jumps
					g.swingOriginX = g.rotation.X
					g.swingOriginZ = g.rotation.Z
					if !g.smoothCubeTransitions {
						g.swingOriginX, g.swingOriginZ = 0, 0
					}
					g.swingAmplitude = 1.0
				case rotationModePulsate:
					// Start pulse phase based on current rotation to avoid jumps
					g.pulsePhase = math.Atan2(g.rotation.Y, g.rotation.X)
				case rotationModeNormal:
					// Continue from current position
					// No reset needed
				}
			}

			// Apply different movements based on mode
			switch g.rotationMode {
			case rotationModeNormal:
				// Standard rotation (existing)
				g.rotation.X += rotationSpeed
				g.rotation.Y += rotationSpeed
				g.rotation.Z -= rotationSpeed

			case rotationModeTumble:
				// Chaotic rotation with changing speed
				speedVar := math.Sin(g.rotationTimer * 0.02)
				g.rotation.X += rotationSpeed * (1 + speedVar)
				g.rotation.Y += rotationSpeed * (1.5 - speedVar*0.5)
				g.rotation.Z -= rotationSpeed * (0.5 + speedVar*0.5)

			case rotationModePulsate:
				// Rotation with pulsation
				g.pulsePhase += 0.05
				pulse := 1.0 + 0.3*math.Sin(g.pulsePhase)
				g.rotation.X += rotationSpeed * pulse
				g.rotation.Y += rotationSpeed * 0.7 * pulse
				g.rotation.Z -= rotationSpeed * 0.3

			case rotationModeSwing:
				// Pendulum swing
				swing := math.Sin(g.rotationTimer*0.03) * g.swingAmplitude
				g.rotation.X = g.swingOriginX + swing*0.8
				g.rotation.Y += rotationSpeed * 0.5
				g.rotation.Z = g.swingOriginZ + swing*0.4
				g.swingAmplitude *= 0.998 // Slower damping

			case rotationModeBounce:
				// Bounce effect
				g.bounceVelocity -= 0.001 // Reduced gravity
				g.bouncePosition += g.bounceVelocity

				// Limit descent to stay visible
				if g.bouncePosition < -0.3 {
					g.bouncePosition = -0.3
					g.bounceVelocity = math.Abs(g.bounceVelocity) * 0.85 // Bounce with energy loss
				}

				g.rotation.X += rotationSpeed * 0.3
				g.rotation.Y += rotationSpeed * (1 + math.Max(0, g.bouncePosition))
				g.rotation.Z += rotationSpeed * 0.1
			}

			// Zoom in 3D cube (existing)
			if g.zoom3d < 1 {
				g.zoom3d += zoomSpeed
				if g.zoom3d > 1 {
					g.zoom3d = 1
				}
			}
		}
	}

	g.cubeTick++
	if g.cubeChanged && g.smoothCubeTransitions {
		g.sampleRawCube(g.cubeIncoming)
		if err := g.cubeHandoff.Begin(g.cubePrevious, g.cubeIncoming, float64(g.cubeTick)/60, .75); err != nil {
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

		// Draw 3D jelly cube
		if g.scrollIteration > 25 {
			g.draw3DCube()
		}

		// Final composite with fade
		g.drawOp.GeoM.Reset()
		g.drawOp.ColorScale.Reset()
		g.drawOp.GeoM.Translate(sceneOffsetX(screen.Bounds().Dx()), 70)
		g.drawOp.ColorScale.ScaleAlpha(float32(g.fadeImg))
		screen.DrawImage(g.stCanvas, g.drawOp)
	}
}

// draw3DCube transforms and draws the jelly cube directly onto the ST canvas.
// sampleRawCube evaluates an effect without reading or advancing render state.
func (g *Game) sampleRawCube(dst []Vector3) {
	// Time factor for animation
	time := g.pos * 3.0

	// Adjust parameters based on mode
	var extraScale float64 = 1.0
	var extraOffsetY float64 = 0
	var twistFactor float64 = 0

	switch g.rotationMode {
	case rotationModePulsate:
		// Pulsing zoom effect
		extraScale = 1.0 + 0.2*math.Sin(g.pulsePhase)
	case rotationModeBounce:
		// Vertical offset for bounce (limited)
		extraOffsetY = g.bouncePosition * 50 // Reduced from 100 to 50
	case rotationModeTumble:
		// Twist effect
		twistFactor = math.Sin(g.rotationTimer*0.01) * 0.5
	}

	// Pre-calculate sin/cos for rotation
	sinX, cosX := math.Sincos(g.rotation.X)
	sinY, cosY := math.Sincos(g.rotation.Y)
	sinZ, cosZ := math.Sincos(g.rotation.Z)

	// Pre-calculate common animation values
	squashFactor := 1.0 + 0.15*math.Sin(time*2.0)
	stretchFactor := 1.0 + 0.15*math.Cos(time*2.0)
	sinTime25, cosTime25 := math.Sincos(time * 2.5)
	secondaryBounce := sinTime25 + 0.5*math.Sin(time*5.0)
	secondaryOffsetY := cosTime25*8.0 + extraOffsetY
	secondaryOffsetZ := math.Sin(time*3.7) * 4.0
	deformScale := 1.0
	if g.rotationMode == rotationModePulsate {
		deformScale += 0.3 * math.Sin(g.pulsePhase*2)
	}

	// Apply transformations to each vertex
	for i, vertex := range g.vertices {
		x, y, z := vertex.X, vertex.Y, vertex.Z

		// Apply extra scale
		x *= extraScale
		y *= extraScale
		z *= extraScale

		// Calculate jelly deformation (existing code)
		positionKey := vertex.X*0.01 + vertex.Y*0.02 + vertex.Z*0.03
		deformAmount := 25.0 * deformScale

		// Multiple wobble frequencies for complex motion
		wobbleX := math.Sin(time+positionKey*5.0) * deformAmount * 0.4
		wobbleX += math.Sin(time*2.1+positionKey*3.0) * deformAmount * 0.2

		wobbleY := math.Cos(time*1.3+positionKey*7.0) * deformAmount * 0.4
		wobbleY += math.Cos(time*1.7+positionKey*4.0) * deformAmount * 0.2

		wobbleZ := math.Sin(time*0.7+positionKey*3.0) * deformAmount * 0.3
		wobbleZ += math.Cos(time*1.9+positionKey*6.0) * deformAmount * 0.15

		// Apply deformation based on distance from center
		distFromCenter := math.Sqrt(x*x+y*y+z*z) / 80.0
		wobbleInfluence := 0.5 + distFromCenter*0.5

		x += wobbleX * wobbleInfluence
		y += wobbleY * wobbleInfluence
		z += wobbleZ * wobbleInfluence

		// Squash and stretch effect
		x *= squashFactor
		y *= stretchFactor
		z *= 1.0 / (squashFactor*stretchFactor*0.5 + 0.5)

		// Add ripple effect
		ripple := math.Sin(time*4.0+distFromCenter*10.0) * 5.0
		x += ripple * (vertex.Y / 80.0)
		y += ripple * (vertex.X / 80.0)

		// Add twist effect if applicable
		if twistFactor != 0 {
			angle := twistFactor * (vertex.Y / 80.0)
			sinAngle, cosAngle := math.Sincos(angle)
			newX := x*cosAngle - z*sinAngle
			newZ := x*sinAngle + z*cosAngle
			x, z = newX, newZ
		}

		// Rotate around X axis
		newY := y*cosX - z*sinX
		newZ := y*sinX + z*cosX
		y, z = newY, newZ

		// Rotate around Y axis
		newX := x*cosY + z*sinY
		newZ = -x*sinY + z*cosY
		x, z = newX, newZ

		// Rotate around Z axis
		newX = x*cosZ - y*sinZ
		newY = x*sinZ + y*cosZ

		// Add secondary wobble
		newX += secondaryBounce * 8.0
		newY += secondaryOffsetY
		newZ += secondaryOffsetZ

		dst[i] = Vector3{X: newX, Y: newY, Z: newZ}
	}

}

func (g *Game) draw3DCube() {
	g.sampleCube(g.transformedVertices)
	// Calculate face depths
	for i, face := range g.faces {
		// Calculate average Z depth
		avgZ := (g.transformedVertices[face.P1].Z + g.transformedVertices[face.P2].Z +
			g.transformedVertices[face.P3].Z + g.transformedVertices[face.P4].Z) / 4.0
		g.facesWithDepth[i].face = face
		g.facesWithDepth[i].depth = avgZ
	}

	// Sort the six faces back to front. An insertion sort avoids the reflection
	// and heap escape caused by sort.Slice in this per-frame hot path.
	for i := 1; i < len(g.facesWithDepth); i++ {
		face := g.facesWithDepth[i]
		j := i
		for j > 0 && g.facesWithDepth[j-1].depth > face.depth {
			g.facesWithDepth[j] = g.facesWithDepth[j-1]
			j--
		}
		g.facesWithDepth[j] = face
	}

	// Draw faces
	centerX := float32(stCanvasWidth / 2)
	centerY := float32(stCanvasHeight / 2)
	g.cubeVertices = g.cubeVertices[:0]
	g.cubeIndices = g.cubeIndices[:0]

	// High FOV for minimal perspective
	fov := 2000.0

	for _, f := range g.facesWithDepth {
		face := f.face

		// Get transformed vertices
		v0 := g.transformedVertices[face.P1]
		v1 := g.transformedVertices[face.P2]
		v2 := g.transformedVertices[face.P3]
		v3 := g.transformedVertices[face.P4]

		// Project to 2D
		offset := 300.0

		scale0 := fov / (fov + v0.Z + offset)
		x0 := centerX + float32(v0.X*scale0)
		y0 := centerY + float32(v0.Y*scale0)

		scale1 := fov / (fov + v1.Z + offset)
		x1 := centerX + float32(v1.X*scale1)
		y1 := centerY + float32(v1.Y*scale1)

		scale2 := fov / (fov + v2.Z + offset)
		x2 := centerX + float32(v2.X*scale2)
		y2 := centerY + float32(v2.Y*scale2)

		scale3 := fov / (fov + v3.Z + offset)
		x3 := centerX + float32(v3.X*scale3)
		y3 := centerY + float32(v3.Y*scale3)

		// Slight expansion to avoid gaps
		expansion := float32(0.5)

		// Calculate face center
		centerFaceX := (x0 + x1 + x2 + x3) / 4.0
		centerFaceY := (y0 + y1 + y2 + y3) / 4.0

		// Expand vertices slightly
		x0 += (x0 - centerFaceX) * expansion / 100.0
		y0 += (y0 - centerFaceY) * expansion / 100.0
		x1 += (x1 - centerFaceX) * expansion / 100.0
		y1 += (y1 - centerFaceY) * expansion / 100.0
		x2 += (x2 - centerFaceX) * expansion / 100.0
		y2 += (y2 - centerFaceY) * expansion / 100.0
		x3 += (x3 - centerFaceX) * expansion / 100.0
		y3 += (y3 - centerFaceY) * expansion / 100.0

		// Apply the former intermediate-canvas zoom directly to the vertices.
		zoom := float32(g.zoom3d)
		x0 = centerX + (x0-centerX)*zoom
		y0 = centerY + (y0-centerY)*zoom
		x1 = centerX + (x1-centerX)*zoom
		y1 = centerY + (y1-centerY)*zoom
		x2 = centerX + (x2-centerX)*zoom
		y2 = centerY + (y2-centerY)*zoom
		x3 = centerX + (x3-centerX)*zoom
		y3 = centerY + (y3-centerY)*zoom

		red := float32(face.Color.R) / 255
		green := float32(face.Color.G) / 255
		blue := float32(face.Color.B) / 255
		alpha := float32(face.Color.A) / 255
		base := uint16(len(g.cubeVertices))
		g.cubeVertices = append(g.cubeVertices,
			ebiten.Vertex{DstX: x0, DstY: y0, SrcX: 0, SrcY: 0, ColorR: red, ColorG: green, ColorB: blue, ColorA: alpha},
			ebiten.Vertex{DstX: x1, DstY: y1, SrcX: 1, SrcY: 0, ColorR: red, ColorG: green, ColorB: blue, ColorA: alpha},
			ebiten.Vertex{DstX: x2, DstY: y2, SrcX: 1, SrcY: 1, ColorR: red, ColorG: green, ColorB: blue, ColorA: alpha},
			ebiten.Vertex{DstX: x3, DstY: y3, SrcX: 0, SrcY: 1, ColorR: red, ColorG: green, ColorB: blue, ColorA: alpha},
		)
		g.cubeIndices = append(g.cubeIndices,
			base, base+1, base+2,
			base, base+2, base+3,
		)
	}

	if len(g.cubeIndices) > 0 {
		g.stCanvas.DrawTriangles(g.cubeVertices, g.cubeIndices, g.whiteImg, nil)
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
	if g.audioPlayer != nil {
		g.audioPlayer.Close()
	}
	if g.ymPlayer != nil {
		g.ymPlayer.Close()
	}
	if g.crtShader != nil {
		g.crtShader.Deallocate()
	}
}

// sampleCube includes any active handoff without depending on Draw calls.
func (g *Game) sampleCube(dst []Vector3) {
	g.sampleRawCube(dst)
	if g.smoothCubeTransitions {
		g.cubeHandoff.Apply(dst, dst, float64(g.cubeTick)/60)
	}
}

// SetSmoothTransitions selects continuous cube effects or the historical path.
// Configure this before the first Update; the DCK version enables it by default.
func (g *Game) SetSmoothTransitions(enabled bool) { g.smoothCubeTransitions = enabled }
