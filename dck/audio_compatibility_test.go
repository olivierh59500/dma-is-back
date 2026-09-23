package dmaisback

import (
	"crypto/sha256"
	"fmt"
	"testing"

	"github.com/olivierh59500/democonstructionkit/sound"
)

// TestMusicPCMCompatibility preserves the audible level and PCM of the original adapter.
func TestMusicPCMCompatibility(t *testing.T) {
	player, err := sound.Open("music.ym", musicData, sound.Options{SampleRate: audioSampleRate, Loop: true, PCMFormat: sound.Float32, Gain: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer player.Close()
	data := make([]byte, audioSampleRate*8)
	for start := 0; start < len(data); {
		end := min(start+4096, len(data))
		n, err := player.Read(data[start:end])
		if err != nil {
			t.Fatal(err)
		}
		if n != end-start {
			t.Fatalf("read %d/%d", n, end-start)
		}
		start = end
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "56f243879bdde2844e7dc094cfd3840ea6657926329a0ccde7ca0ec1e9859c6b" {
		t.Fatalf("soundtrack PCM changed: %s", got)
	}
}
