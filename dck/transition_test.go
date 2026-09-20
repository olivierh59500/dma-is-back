package dmaisback

import (
	"math"
	"testing"
)

func TestEveryCubeModeStartsAtThePreviousPose(t *testing.T) {
	for mode := 0; mode < rotationModeTotal; mode++ {
		g := NewGame()
		g.audioReady = true
		g.introComplete = true
		g.scrollIteration = 100
		g.zoom3d = 1
		g.rotationMode = mode
		g.rotationTimer = g.rotationDuration - 1
		g.rotation = Vector3{1.3, 2.7, -.9}
		g.pos = 23.71
		g.pulsePhase = 2.1
		g.bouncePosition = -.2
		before, after, raw := make([]Vector3, len(g.vertices)), make([]Vector3, len(g.vertices)), make([]Vector3, len(g.vertices))
		g.sampleCube(before)
		if err := g.Update(); err != nil {
			t.Fatal(err)
		}
		g.sampleCube(after)
		for i := range before {
			if after[i] != before[i] {
				t.Fatalf("mode %d vertex %d jumped: %v -> %v", mode, i, before[i], after[i])
			}
		}
		// Drawing frequency must not affect the pose or handoff clock.
		g.sampleCube(raw)
		for i := range raw {
			if raw[i] != after[i] {
				t.Fatal("sampling advanced the animation")
			}
		}
		for frame := 0; frame < 46; frame++ {
			copy(before, after)
			if err := g.Update(); err != nil {
				t.Fatal(err)
			}
			g.sampleCube(after)
			for i := range after {
				v := after[i].Sub(before[i])
				if math.Sqrt(v.Dot(v)) > 30 {
					t.Fatalf("large transition step: mode %d frame %d", mode, frame)
				}
			}
		}
		g.sampleRawCube(raw)
		for i := range raw {
			if raw[i] != after[i] {
				t.Fatal("handoff correction did not expire")
			}
		}
		g.Cleanup()
	}
}
