# DMA is Back! - Atari ST Demo Remake

<div align="center">
  
![Go Version](https://img.shields.io/badge/Go-1.25%2B-00ADD8?style=for-the-badge&logo=go)
![Ebiten](https://img.shields.io/badge/Ebitengine-v2.9.11-FF6B6B?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Desktop%20|%20Android%20|%20Web-4EAA25?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**A modern Go/Ebiten remake of a classic Atari ST demoscene production**

[Live Demo](#live-demo) • [Features](#features) • [Installation](#installation) • [Building](#building) • [Credits](#credits)

</div>

---

## 🎮 Overview

This is a tribute to the golden era of the Atari ST demoscene, reimagined using modern technologies. The demo features the iconic "jelly cube" effect combined with parallax distortion scrolling, bringing classic 16-bit demo effects to contemporary platforms.

Originally inspired by demos from The Carebears (TCB), this remake showcases how timeless these visual effects remain, now running smoothly in your web browser or as a native application.

## ✨ Features

### Visual Effects
- **3D Jelly Cube**: A mesmerizing, wobbling 3D cube with dynamic deformation
- **Parallax Distortion Scrolling**: Text that flows with sinusoidal wave distortions
- **CRT Shader Effect**: Authentic retro monitor simulation with:
  - Barrel distortion
  - Scanlines
  - RGB chromatic aberration
  - Vignette effect
- **Animated Logo Background**: Smoothly moving tiled patterns

### Audio
- **YM Chiptune Playback**: Authentic Atari ST sound using the YM2149 emulation
- **Looped Music**: Classic demoscene soundtrack that brings back the nostalgia

### Technical Features
- **Cross-platform**: Runs on Windows, macOS, Linux, Android, and Web browsers (via WebAssembly)
- **Optimized Performance**: Pre-allocated buffers and efficient rendering
- **60 FPS**: Smooth animations matching modern display standards

## 🚀 Live Demo

Experience the demo directly in your browser: [Coming Soon]

## 📋 Requirements

- Go 1.25 or higher
- For native builds: OpenGL support
- For web builds: Modern browser with WebAssembly support
- For Android builds: Android SDK 36, NDK 28.2, JDK 17, and an arm64 device

## 🔧 Installation

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/olivierh59500/dma-is-back.git
cd dma-is-back
```

2. Install dependencies:
```bash
go mod download
```

3. Run the demo:
```bash
go run ./cmd/dmaisback
```

### Building from Source

#### Native Build
```bash
# Windows
GOOS=windows GOARCH=amd64 go build -o dma-demo.exe ./cmd/dmaisback

# macOS
GOOS=darwin GOARCH=amd64 go build -o dma-demo ./cmd/dmaisback

# Linux
GOOS=linux GOARCH=amd64 go build -o dma-demo ./cmd/dmaisback
```

#### Android (Pixel / arm64)

With USB debugging enabled and exactly one authorized Android device connected:

```bash
./scripts/run-android.sh
```

The script generates the Ebitengine AAR, builds a debug APK, installs it, and
launches `com.olivierh.dmaisback/.MainActivity`. It uses the validated toolchain:
Ebitengine 2.9.11, Gradle 8.11.1, Android Gradle Plugin 8.10.1, SDK 36,
NDK 28.2, and JDK 17.

#### WebAssembly Build
```bash
# Copy wasm_exec.js
cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" .

# Build WASM binary
GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o demo.wasm ./cmd/dmaisback

# Serve locally
python3 -m http.server 8080
# Open http://localhost:8080 in your browser
```

## 📁 Project Structure

```
dma-is-back/
├── game.go                 # Shared demo implementation
├── cmd/dmaisback/          # Desktop/Web entry point
├── mobile/                 # ebitenmobile bridge
├── android/                # Native Android shell and Gradle wrapper
├── scripts/run-android.sh  # Build, install, and launch on an arm64 device
├── assets/                 # Demo resources
│   ├── font.png           # Bitmap font (Atari ST style)
│   ├── small-dma-jelly.png # Logo/background image
│   └── Mindbomb.ym        # YM chiptune music
├── index.html             # Web deployment HTML
├── wasm_exec.js           # Go WASM support file
├── go.mod                  # Go module file
└── README.md              # This file
```

## 🎨 Demo Sections

### 1. Intro Sequence
- Scrolling text with CRT shader effect
- Classic demoscene message: "IF YOU THINK THIS IS ALL, YOU'RE SO WRONG..."

### 2. Main Demo
- **Background**: Animated tiled logo with sinusoidal movement
- **Middle Layer**: Distorted scrolling text with multiple wave effects
- **Foreground**: 3D jelly cube with complex deformation animations

## 🛠️ Technical Details

### Graphics Engine
- **Ebiten v2**: Hardware-accelerated 2D game engine
- **Custom Shaders**: GLSL-style shaders compiled for Ebiten
- **Resolution**: 640x400 Atari ST canvas in a 768x540 logical viewport
- **Wide screens**: Preserves the demo aspect ratio and centers it instead of stretching it

### Audio System
- **YM Player**: Accurate YM2149 sound chip emulation
- **Format**: YM files (compressed Atari ST music format)

### Optimization Techniques
- Pre-allocated vertex and face buffers
- Reusable draw options to minimize GC pressure
- Efficient sorting algorithm for depth ordering
- Cached text rendering for scrolling effects

## 🙏 Credits

### Original Concept
- **The 24H Demo (TCB)**: Original Atari ST demo effects
- **Megadist Demo (ULM)**: Distortion scrolling inspiration

### Remake
- **Code**: [olivierh59500](https://github.com/olivierh59500/dma-is-back)
- **Framework**: [Hajime Hoshi](https://github.com/hajimehoshi) for Ebiten
- **YM Player**: [olivierh59500](https://github.com/olivierh59500/ym-player)

### Music
- YM chiptune from the Atari ST demoscene era (in the "Mindbomb" demo from TLB)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Share your memories of the Atari ST era

## 🌟 Acknowledgments

Special thanks to the entire Atari ST demoscene community for pushing the boundaries of what was possible on 16-bit hardware. This remake is a tribute to your creativity and technical excellence.

---

<div align="center">
  
**Remember: DMA is back!**

Made with ❤️ and lots of nostalgia

</div>

## Optional DCK version

The original implementation remains at its original paths. Run it with `go run ./cmd/dmaisback`.

The construction-kit version is in [dck/](dck/README.md). Run `go run ./dck/cmd/dmaisback` from this directory. Both versions share the original assets.
