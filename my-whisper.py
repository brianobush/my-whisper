import os
import sys
import signal
import shutil
import subprocess
import threading
from datetime import datetime

if sys.platform != "darwin":
    sys.exit("This tool only runs on macOS.")

import pynput.keyboard as keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper

AFPLAY = shutil.which("afplay")
PBCOPY = shutil.which("pbcopy")
if not AFPLAY or not PBCOPY:
    sys.exit("Required macOS utilities (afplay, pbcopy) not found.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(SCRIPT_DIR, "sounds")
START_SOUND = os.path.join(SOUNDS_DIR, "start.wav")
END_SOUND = os.path.join(SOUNDS_DIR, "end.wav")

for _snd in (START_SOUND, END_SOUND):
    if not os.path.isfile(_snd):
        sys.exit(f"Missing sound file: {_snd}")


def play_sound(path):
    if not os.path.isfile(path):
        return
    try:
        subprocess.Popen([AFPLAY, path])
    except Exception as e:
        print(f"Could not play sound {path}: {e}")


def copy_to_clipboard(text):
    try:
        process = subprocess.Popen([PBCOPY], stdin=subprocess.PIPE)
        process.communicate(text.encode("utf-8"))
    except Exception as e:
        print(f"Could not copy to clipboard: {e}")


class HotkeyHandler:
    def __init__(self):
        self.hotkeys = {}
        self.current_keys = set()
        self.listener = None

    def add_hotkey(self, hotkey_str, callback):
        pynput_keys = frozenset(
            keyboard.HotKey.parse(self._convert(hotkey_str))
        )
        self.hotkeys[pynput_keys] = callback

    def on_press(self, key):
        try:
            canonical = self.listener.canonical(key)
            self.current_keys.add(canonical)
            for combo, cb in self.hotkeys.items():
                if combo.issubset(self.current_keys):
                    cb()
                    break
        except Exception:
            pass

    def on_release(self, key):
        try:
            canonical = self.listener.canonical(key)
            self.current_keys.discard(canonical)
        except Exception:
            pass

    def start(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release, suppress=False
        )
        self.listener.daemon = True
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None

    @staticmethod
    def _convert(hotkey):
        parts = hotkey.lower().split("+")
        converted = []
        for part in parts:
            part = part.strip()
            if part in ("cmd", "command"):
                converted.append("<cmd>")
            elif part == "option":
                converted.append("<alt>")
            elif part == "control":
                converted.append("<ctrl>")
            elif len(part) > 1:
                converted.append(f"<{part}>")
            else:
                converted.append(part)
        return "+".join(converted)


class Recorder:
    def __init__(self, max_duration=30):
        self.recording = False
        self.max_duration = max_duration
        self.stop_event = threading.Event()
        self.chunks = []
        self.stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.chunks.append(indata.copy())

    def record(self):
        self.recording = True
        self.stop_event.clear()
        self.chunks.clear()

        try:
            device_info = sd.query_devices(sd.default.device, "input")
            samplerate = int(device_info["default_samplerate"])

            self.stream = sd.InputStream(
                callback=self._callback, channels=1, samplerate=samplerate
            )
            with self.stream:
                print("Recording...")
                self.stop_event.wait(timeout=self.max_duration)
        except Exception as e:
            print(f"Recording error: {e}")
            return None
        finally:
            self.recording = False
            self.stream = None

        if not self.chunks:
            print("No audio captured.")
            return None

        wav_path = os.path.join(
            "/tmp", f"whisper_asr_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.wav"
        )
        try:
            with sf.SoundFile(
                wav_path, mode="x", samplerate=samplerate,
                channels=1, subtype="PCM_16"
            ) as f:
                for chunk in self.chunks:
                    f.write(chunk)
            return wav_path
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def stop(self):
        if self.recording:
            self.stop_event.set()


class WhisperASR:
    def __init__(self, model_name="base", hotkey="ctrl+alt+r", max_duration=30):
        print(f"Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        print("Model loaded.")

        self.recorder = Recorder(max_duration=max_duration)
        self.hotkey_handler = HotkeyHandler()
        self.hotkey_handler.add_hotkey(hotkey, self._on_hotkey)

        self.recording_thread = None
        self.shutdown_event = threading.Event()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\nShutting down...")
        self.shutdown_event.set()
        self.stop()

    def _on_hotkey(self):
        if not self.recorder.recording:
            play_sound(START_SOUND)
            print("Hotkey: start recording")
            self.recording_thread = threading.Thread(
                target=self._record_and_transcribe, daemon=True
            )
            self.recording_thread.start()
        else:
            print("Hotkey: stop recording")
            self.recorder.stop()

    def _record_and_transcribe(self):
        audio_path = self.recorder.record()
        play_sound(END_SOUND)
        if not audio_path:
            return

        print("Transcribing...")
        try:
            result = self.model.transcribe(audio_path, language="en", fp16=False)
            text = " ".join(seg["text"].strip() for seg in result["segments"])
            print(f"Transcription: {text}")

            if text.strip():
                copy_to_clipboard(text)
                print("Copied to clipboard.")
        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    def run(self):
        self.hotkey_handler.start()
        print("Whisper ASR ready. Press Ctrl+Alt+R to start/stop recording.")
        print("Press Ctrl+C to quit.")
        try:
            while not self.shutdown_event.is_set():
                self.shutdown_event.wait(0.1)
        finally:
            self.stop()

    def stop(self):
        self.recorder.stop()
        self.hotkey_handler.stop()
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1)
        sd.stop()


if __name__ == "__main__":
    asr = WhisperASR(model_name="base", hotkey="ctrl+alt+r", max_duration=30)
    try:
        asr.run()
    except KeyboardInterrupt:
        pass
