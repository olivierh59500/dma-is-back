//go:build dck_rendercheck

package dmaisback

import (
	"bytes"
	"fmt"
	"os"
	"testing"

	"github.com/hajimehoshi/ebiten/v2"
	capture "github.com/olivierh59500/democonstructionkit/fidelity/ebiten"
	"github.com/olivierh59500/democonstructionkit/geometry"
	"github.com/olivierh59500/democonstructionkit/render"
)

type deformationRenderCheck struct {
	game               *Game
	frame, checked     int
	previous, incoming []Vector3
	handoff            geometry.Handoff
	actual, expected   *ebiten.Image
	pixelsA, pixelsB   []byte
	err                error
}

var cubeCheckFrames = []int{0, 1, 90, 298, 299, 300, 301, 321, 343, 345, 390, 598, 599, 600, 601, 621, 643, 645, 690, 898, 899, 900, 901, 921, 943, 945, 990, 1198, 1199, 1200, 1201, 1221, 1243, 1245, 1290, 1498, 1499, 1500, 1501, 1521, 1543, 1545}

func (c *deformationRenderCheck) Layout(int, int) (int, int) { return stCanvasWidth, stCanvasHeight }
func (c *deformationRenderCheck) sampleReference(dst []Vector3) {
	c.game.sampleRawCubeReference(dst)
	c.handoff.Apply(dst, dst, float64(c.game.cubeTick)/60)
}
func (c *deformationRenderCheck) Update() error {
	if c.err != nil {
		return c.err
	}
	c.sampleReference(c.previous)
	if err := c.game.Update(); err != nil {
		return err
	}
	if c.game.cubeChanged {
		c.game.sampleRawCubeReference(c.incoming)
		if err := c.handoff.Begin(c.previous, c.incoming, float64(c.game.cubeTick)/60, .75); err != nil {
			return err
		}
	}
	c.frame++
	return nil
}
func (c *deformationRenderCheck) Draw(dst *ebiten.Image) {
	if c.err != nil || c.checked >= len(cubeCheckFrames) || c.frame != cubeCheckFrames[c.checked] {
		return
	}
	c.actual.Clear()
	c.expected.Clear()
	c.game.stCanvas = c.actual
	c.game.draw3DCube()

	c.sampleReference(c.game.transformedVertices)
	c.game.stCanvas = c.expected
	c.game.drawCubeVertices()
	c.actual.ReadPixels(c.pixelsA)
	c.expected.ReadPixels(c.pixelsB)
	if !bytes.Equal(c.pixelsA, c.pixelsB) {
		different := 0
		for i := 0; i < len(c.pixelsA); i += 4 {
			if !bytes.Equal(c.pixelsA[i:i+4], c.pixelsB[i:i+4]) {
				different++
			}
		}
		c.err = fmt.Errorf("cube frame %d mode %d changed %d pixels", c.frame, c.game.rotationMode, different)

	}
	dst.DrawImage(c.actual, nil)
	c.checked++
}
func TestMain(m *testing.M) {
	if code := m.Run(); code != 0 {
		os.Exit(code)
	}
	dir, err := os.MkdirTemp("", "dma-deformation-")
	if err != nil {
		panic(err)
	}
	var check *deformationRenderCheck
	err = capture.Run(capture.Config{Directory: dir, Frames: cubeCheckFrames, Width: stCanvasWidth, Height: stCanvasHeight}, func() (ebiten.Game, error) {
		g := NewGame()
		g.audioReady = true
		g.introComplete = true
		g.scrollIteration = 100
		g.zoom3d = 1
		check = &deformationRenderCheck{game: g, previous: make([]Vector3, len(g.vertices)), incoming: make([]Vector3, len(g.vertices)), actual: render.NewSurface(stCanvasWidth, stCanvasHeight), expected: render.NewSurface(stCanvasWidth, stCanvasHeight), pixelsA: make([]byte, stCanvasWidth*stCanvasHeight*4), pixelsB: make([]byte, stCanvasWidth*stCanvasHeight*4)}
		return check, nil
	})
	if err == nil && check != nil {
		err = check.err
		if err == nil && check.checked != len(cubeCheckFrames) {
			err = fmt.Errorf("only %d cube captures checked", check.checked)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("All %d cube captures match exactly: %s\n", check.checked, dir)
}
