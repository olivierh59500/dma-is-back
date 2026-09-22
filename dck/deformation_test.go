package dmaisback

import (
	"math"
	"testing"
)

// This independent reference retains the original arithmetic as a fidelity oracle.
func (g *Game) sampleRawCubeReference(dst []Vector3) {
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

func TestSharedCubeDeformationMatchesOriginalModes(t *testing.T) {
	g := &Game{vertices: []Vector3{{-80, -80, -80}, {80, -80, -80}, {80, 80, -80}, {-80, 80, -80}, {-80, -80, 80}, {80, -80, 80}, {80, 80, 80}, {-80, 80, 80}, {-27.25, 19.75, 41.125}, {0, 0, 0}}}
	var want, got [10]Vector3
	for mode := 0; mode < rotationModeTotal; mode++ {
		g.rotationMode = mode
		for frame := 0; frame < 601; frame++ {
			g.pos = float64(frame) * .014
			g.rotation = Vector3{float64(frame) * .05, float64(frame) * .037, -float64(frame) * .019}
			g.rotationTimer = float64(frame % 300)
			g.pulsePhase = float64(frame) * .05
			g.bouncePosition = math.Sin(float64(frame)*.043) * .3
			g.sampleRawCubeReference(want[:])
			g.sampleRawCube(got[:])
			for i := range got {
				if delta := got[i].Sub(want[i]); math.Max(math.Abs(delta.X), math.Max(math.Abs(delta.Y), math.Abs(delta.Z))) > 1e-11 {
					t.Fatalf("mode %d frame %d vertex %d: got %#v, want %#v", mode, frame, i, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkSharedCubeDeformation(b *testing.B) {
	g := &Game{vertices: []Vector3{{-80, -80, -80}, {80, -80, -80}, {80, 80, -80}, {-80, 80, -80}, {-80, -80, 80}, {80, -80, 80}, {80, 80, 80}, {-80, 80, 80}}, pos: 2.3, rotation: Vector3{1, 2, 3}, rotationMode: rotationModeTumble, rotationTimer: 150}
	var dst [8]Vector3
	g.sampleRawCube(dst[:])
	b.ReportAllocs()
	for b.Loop() {
		g.sampleRawCube(dst[:])
	}
}
