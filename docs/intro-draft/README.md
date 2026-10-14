# Apertus intro — creative draft

118-second, 1920×1080 English narrated introduction for the OST Track 2A jury. Draft direction and script developed by the explicitly requested Astra agent; source verification, local voice generation and rendering completed by the primary agent.

[Watch or download the MP4](apertus-intro-draft.mp4?raw=true). English captions are burned into the image and included as a selectable subtitle track. [SRT](subtitles.en.srt), [storyboard](storyboard.md), [narration](narration.txt), [scene data](scenes.json) and [render provenance](render-manifest.json) remain editable.

## Rebuild

Requires Python 3.12, FFmpeg with libx264/libass, and DejaVu Sans fonts. From the repository root:

```bash
python -m venv .venv-intro
.venv-intro/bin/pip install -r docs/intro-draft/requirements.txt
mkdir -p /tmp/apertus-intro-voice /tmp/apertus-intro-build
curl -fL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o /tmp/apertus-intro-voice/kokoro-v1.0.onnx
curl -fL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o /tmp/apertus-intro-voice/voices-v1.0.bin
.venv-intro/bin/python docs/intro-draft/synthesize.py --model-dir /tmp/apertus-intro-voice --build-dir /tmp/apertus-intro-build
.venv-intro/bin/python docs/intro-draft/render.py --build-dir /tmp/apertus-intro-build
```

The synthesizer checks fixed model/voice SHA-256 hashes before use. Model weights and intermediate WAV/video files remain outside the repo. Downloaded models are about 354 MB. No GPU rental, API key or voice service is needed. The synthetic narrator uses Kokoro v1.0 `af_heart` (American English), locally via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx). [Kokoro model source and license](https://huggingface.co/hexgrad/Kokoro-82M). No human speaker was recorded or impersonated. Captions follow sentence audio timing with proportionally timed word groups within sentences; they are readable phrase captions, not forced word alignment.

## Scientific scope

All reported scores come from [the technical report](../../track_2a/technical_report.md): 276 development-validation examples. Base full/capped .671612 and original full/capped head .894461 share that context; selected hybrid-8k .949069 is later development with a different context and a context-matched head. Cross-language .941796 is a subset metric. Neither is an organizer-hidden score or an independent final-system test. The final 8k system needs evaluation on new independent booklets.

14,106.1 → 8,059.6 mean prompt tokens is a 42.864% reduction, rounded to 43%; it is not a measured latency or cost reduction. All Apertus weights stay frozen; only the lightweight logistic-regression decision head is trained. No LoRA is claimed. Source passages are exact input-context candidates, not validated minimal evidence.

“Jev-style” describes the user-supplied starting-point inspiration. The repo does not establish a primary Jev reference; the draft does not claim to reproduce Jev or assign its published results to our work. The storyboard documents this attribution limit.

The embedded application screenshot comes from the already verified [real English browser session](../../docs/demo/README.md); this intro is an explanatory animation, not another GPU inference recording. No internal holdout was accessed or new benchmark performed for this video.
