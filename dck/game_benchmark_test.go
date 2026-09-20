package dmaisback

import "testing"

func benchmarkYMRead(b *testing.B, frames int) {
	player, err := NewYMPlayer(musicData, audioSampleRate, true)
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

func BenchmarkYMRead192Frames(b *testing.B) {
	benchmarkYMRead(b, 192)
}

func BenchmarkYMRead2048Frames(b *testing.B) {
	benchmarkYMRead(b, 2048)
}

func BenchmarkYMReadFloat32_192Frames(b *testing.B) {
	player, err := NewYMPlayer(musicData, audioSampleRate, true)
	if err != nil {
		b.Fatal(err)
	}
	defer player.Close()

	buf := make([]byte, 192*8)
	b.SetBytes(int64(len(buf)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := player.readFloat32(buf); err != nil {
			b.Fatal(err)
		}
	}
}
