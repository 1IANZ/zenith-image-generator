# Changelog

## [0.3.0] - 2025-12-06

### Added
- Multiple API providers: Gitee AI, HF Z-Image Turbo, HF Qwen Image
- HF Spaces API endpoints (`/api/generate-hf`, `/api/upscale`)
- 4x image upscaling via RealESRGAN
- Floating toolbar with blur, info, download, delete actions
- Image info panel showing size, API provider, upscale status
- HF Token support for extra quota

### Changed
- API provider dropdown with frosted glass effect (backdrop-blur)
- UHD switch styling improvements

## [0.2.0] - 2025-12-05

### Added
- Redesigned image generator UI
- Settings persistence to localStorage
- Last generated image persistence

### Changed
- UI polish for switch, slider and prompt textarea

## [0.1.0] - 2025-12-04

### Added
- Dark mode Gradio-style UI
- Text-to-image generation via Gitee AI API
- Multiple aspect ratio presets (1:1, 16:9, 9:16, 4:3, 3:4)
- Adjustable inference steps and dimensions
- AES-256-GCM encryption for API key storage
- Cloudflare Pages deployment support
- Chinese README documentation
