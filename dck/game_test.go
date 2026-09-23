package dmaisback

import (
	"math"
	"testing"

	"github.com/olivierh59500/democonstructionkit/sound"

	"github.com/hajimehoshi/ebiten/v2"
)

func TestLogicalWidth(t *testing.T) {
	tests := []struct {
		name          string
		outsideWidth  int
		outsideHeight int
		want          int
	}{
		{name: "unknown size", want: screenWidth},
		{name: "original ratio", outsideWidth: 768, outsideHeight: 540, want: 768},
		{name: "portrait", outsideWidth: 1080, outsideHeight: 2424, want: 768},
		{name: "Pixel 10a landscape", outsideWidth: 2424, outsideHeight: 1080, want: 1212},
		{name: "ultrawide clamp", outsideWidth: 4000, outsideHeight: 540, want: 1280},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := logicalWidth(test.outsideWidth, test.outsideHeight); got != test.want {
				t.Fatalf("logicalWidth(%d, %d) = %d, want %d", test.outsideWidth, test.outsideHeight, got, test.want)
			}
		})
	}
}

func TestSceneOffsetX(t *testing.T) {
	if got, want := sceneOffsetX(screenWidth), 64.0; got != want {
		t.Fatalf("sceneOffsetX(screenWidth) = %v, want %v", got, want)
	}
	if got, want := sceneOffsetX(1212), 286.0; got != want {
		t.Fatalf("sceneOffsetX(Pixel 10a width) = %v, want %v", got, want)
	}
}

func TestWaveByStripMatchesDirectLookup(t *testing.T) {
	g := &Game{frontMainWave: []int{2, 5, 4, 8}}
	for _, start := range []int{0, 1, 3, 4, 11, 127} {
		g.frontWavePos = start
		g.updateWaveByStrip()
		for strip, got := range g.waveByStrip {
			want := g.getSum(g.frontMainWave, start+strip, 0)
			if got != want {
				t.Fatalf("start %d, strip %d: got %d, want %d", start, strip, got, want)
			}
		}
	}
}

func TestGroupedScrollRowsMatchOriginalMapping(t *testing.T) {
	for bounce := 0; bounce < fontHeight; bounce++ {
		for strip := 0; strip < scrollStrips; strip++ {
			dstY := strip * scrollStripH
			height := min(scrollStripH, stCanvasHeight-dstY)
			srcY := ((strip + bounce) % fontHeight) * scrollStripH
			for row := 0; row < height; row++ {
				line := dstY + row
				want := ((line/scrollStripH+bounce)%fontHeight)*scrollStripH + line%scrollStripH
				if got := srcY + row; got != want {
					t.Fatalf("bounce %d, line %d: got %d, want %d", bounce, line, got, want)
				}
			}
		}
	}
}

func TestAppendScrollQuad(t *testing.T) {
	g := &Game{
		scrollVertices: make([]ebiten.Vertex, 0, 4),
		scrollIndices:  make([]uint16, 0, 6),
	}
	g.appendScrollQuad(10, 20, 30, 40, 50, 3)

	if len(g.scrollVertices) != 4 || len(g.scrollIndices) != 6 {
		t.Fatalf("got %d vertices and %d indices", len(g.scrollVertices), len(g.scrollIndices))
	}
	if got, want := g.scrollVertices[2].DstX, float32(60); got != want {
		t.Fatalf("bottom-right DstX = %v, want %v", got, want)
	}
	if got, want := g.scrollVertices[2].SrcY, float32(43); got != want {
		t.Fatalf("bottom-right SrcY = %v, want %v", got, want)
	}
	wantIndices := [...]uint16{0, 1, 2, 0, 2, 3}
	for i, want := range wantIndices {
		if got := g.scrollIndices[i]; got != want {
			t.Fatalf("index %d = %d, want %d", i, got, want)
		}
	}
}

func TestMusicFloat32ReaderMatchesInt16(t *testing.T) {
	const frames = 256
	intPlayer, err := sound.Open("music.ym", musicData, sound.Options{SampleRate: audioSampleRate, Loop: true, PCMFormat: sound.PCM16, Gain: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer intPlayer.Close()
	floatPlayer, err := sound.Open("music.ym", musicData, sound.Options{SampleRate: audioSampleRate, Loop: true, PCMFormat: sound.Float32, Gain: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer floatPlayer.Close()

	intData := make([]byte, frames*4)
	floatData := make([]byte, frames*8)
	if n, err := intPlayer.Read(intData); err != nil || n != len(intData) {
		t.Fatalf("int16 read = (%d, %v), want (%d, nil)", n, err, len(intData))
	}
	if n, err := floatPlayer.Read(floatData); err != nil || n != len(floatData) {
		t.Fatalf("float32 read = (%d, %v), want (%d, nil)", n, err, len(floatData))
	}

	for frame := 0; frame < frames; frame++ {
		intOffset := frame * 4
		sample := int16(uint16(intData[intOffset]) | uint16(intData[intOffset+1])<<8)
		want := float32(sample) / (1 << 15)
		floatOffset := frame * 8
		leftBits := uint32(floatData[floatOffset]) |
			uint32(floatData[floatOffset+1])<<8 |
			uint32(floatData[floatOffset+2])<<16 |
			uint32(floatData[floatOffset+3])<<24
		rightBits := uint32(floatData[floatOffset+4]) |
			uint32(floatData[floatOffset+5])<<8 |
			uint32(floatData[floatOffset+6])<<16 |
			uint32(floatData[floatOffset+7])<<24
		if left, right := math.Float32frombits(leftBits), math.Float32frombits(rightBits); left != want || right != want {
			t.Fatalf("frame %d = (%v, %v), want (%v, %v)", frame, left, right, want, want)
		}
	}
}
