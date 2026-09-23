package dmaisback

import (
	"testing"

	"github.com/olivierh59500/democonstructionkit/sound"
)

func benchmarkMusicRead(b *testing.B, frames int) {
	player, err := sound.Open("music.ym", musicData, sound.Options{SampleRate: audioSampleRate, Loop: true, PCMFormat: sound.PCM16, Gain: 1})
	if err != nil {
		b.Fatal(err)
	}
	defer player.Close()

	buf := make([]byte, frames*4)
	b.SetBytes(int64(len(buf)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := player.Read(buf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMusicRead192Frames(b *testing.B) {
	benchmarkMusicRead(b, 192)
}

func BenchmarkMusicRead2048Frames(b *testing.B) {
	benchmarkMusicRead(b, 2048)
}

func BenchmarkMusicReadFloat32_192Frames(b *testing.B) {
	player, err := sound.Open("music.ym", musicData, sound.Options{SampleRate: audioSampleRate, Loop: true, PCMFormat: sound.Float32, Gain: 1})
	if err != nil {
		b.Fatal(err)
	}
	defer player.Close()

	buf := make([]byte, 192*8)
	b.SetBytes(int64(len(buf)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := player.Read(buf); err != nil {
			b.Fatal(err)
		}
	}
}
