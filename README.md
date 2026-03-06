# My Whisper ASR speech to text 

This is a simple tool that I use quite often for transcribing my rambling speech into text. 
I originally had this code running in a single agent, but I quickly grew out of it since I found
myself going between Cursor, Claude Code and ChatGPT. Since this was made for me, it is a mac OS tool only. 
Press a hotkey to record, press again to stop. The transcription is copied to your clipboard.

## Requirements

- macOS
- Python 3.10+

## Setup

```bash
./init.sh
```

This creates a virtual environment and installs dependencies.

## Usage

```bash
source ./env/bin/activate
python my-whisper.py
```

- **Ctrl+Alt+R** — start/stop recording
- **Ctrl+C** — quit

Audio feedback plays `start.wav` and `end.wav` on record start/stop. Transcribed text is automatically copied to the paste buffer.

## Configuration

Edit the `__main__` block in `whisper_asr.py`:

- `model_name` — Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
- `hotkey` — key combination (e.g. `ctrl+alt+r`, `cmd+shift+s`)
- `max_duration` — max recording length in seconds



