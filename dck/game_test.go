package dmaisback

import (
	"math"
	"testing"

	"github.com/olivierh59500/democonstructionkit/sound"
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
