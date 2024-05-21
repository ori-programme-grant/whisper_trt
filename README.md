# WhisperTRT

This project optimizes [OpenAI Whisper](https://github.com/openai/whisper) with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt#:~:text=NVIDIA%20TensorRT%2DLLM%20is%20an,on%20the%20NVIDIA%20AI%20platform.), enabling lower memory consumption, and higher throughput compared to running Whisper with PyTorch.

WhisperTRT roughly mimics the API of the original Whisper model, making it easy to use.  Read below for performance and usage details.

## Benchmark

All benchmarks are generated by calling ``profile_backends.py``,
processing a 20 second audio clip.


### Execution Time

Execution time in seconds to transcribe 20 seconds of speech on Jetson Orin Nano. See [profile_backends.py](profile_backends.py) for details.


|     | whisper | faster_whisper | whisper_trt |
|-------|---------|--------------------|--------|
| tiny.en | 1.74 sec | 0.85 sec | **0.64 sec** |
| base.en | 2.55 sec | Unavailable | **0.86 sec** |


### Memory Consumption

Memory consumption to transcribe 20 seconds of speech on Jetson Orin Nano. See [profile_backends.py](profile_backends.py) for details.

|     | whisper | faster_whisper | whisper_trt |
|-------|---------|--------------------|--------|
| tiny.en | 569 MB | **404 MB** | 488 MB |
| base.en | 666 MB |  Unavailable | **439 MB** |

## Usage

### Python

```python3
from whisper_trt import load_trt_model

model = load_trt_model("tiny.en")

result = model.transcribe("speech.wav") # or pass numpy array

print(result['text'])
```

> You can download an example speech file from [here](https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav) or 
> ``wget https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav -O speech.wav``.


### Transcribe

This script simply runs the model once.  

> Please note:  The first time you call load_model, it takes some time to build the TensorRT engine.
> After the first run, the model will be cached in the directory ~/.cache/whisper_trt/.

```bash
python examples/transcribe.py tiny.en assets/speech.wav --backend whisper_trt
```

### Profile Backend

This scripts measures the latency and process memory when transcribing audio. It includes a warmup run for
more accurate timing.

```bash
python examples/profile_backend.py tiny.en assets/speech.wav --backend whisper_trt
```

Backend can be one of "whisper_trt", "whisper", or "faster_whisper".

### Live Transcription

This script demonstrates live transcription using a microphone and voice activity detection.

```bash
python examples/live_transcription.py tiny.en --backend whisper_trt
```

## See also

- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) - Used to convert PyTorch model to TensorRT and perform inference.
- [NanoLLM](https://github.com/dusty-nv/NanoLLM) - Large Language Models targeting NVIDIA Jetson.  Perfect for combining with ASR!