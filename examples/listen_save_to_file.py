import numpy as np
import time
import pyaudio
import wave
from multiprocessing import Process, Queue, Event
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
from whisper_trt.model import load_trt_model

class ASR:

    def __init__(self, model: str):
        super().__init__()
        self.model_name = model

    def warmup(self):
        from whisper_trt import load_trt_model
        self.model = load_trt_model(self.model_name)

        # warmup
        self.model.transcribe(np.zeros(1536, dtype=np.float32))


    def run(self, audio):

        # Transcribe
        t0 = time.perf_counter_ns()
        text = self.model.transcribe(audio)['text']
        t1 = time.perf_counter_ns()

        print(f"Text: {text}")
        print(f"Transcription Time: {(t1 - t0) / 1e9:.2f} seconds")


def find_respeaker_audio_device_index():
    p = pyaudio.PyAudio()
    device_index = None

    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")

    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if "3dsa" in device_info.get("name").lower():
            device_index = i

    p.terminate()

    if device_index is None:
        raise RuntimeError("ReSpeaker device not found!")

    return device_index

def record_audio(device_index, filename, model, duration=10, rate=44100, channels=1):
    chunk_size = 8192
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size
    )

    print(f"Recording for {duration} seconds...")
    frames = []

    for _ in range(0, int(rate / chunk_size * duration)):
        data = stream.read(chunk_size, exception_on_overflow=True)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    #
    filename = 'output.wav'
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()
    model.run(filename)


if __name__ == "__main__":
    device_index = find_respeaker_audio_device_index()
    model = ASR('base.en')
    model.warmup()
    record_audio(device_index, "output.wav", model, duration=5)
