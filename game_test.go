package dmaisback

import "testing"

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
