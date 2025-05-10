# Prerequisite: pip install -e   inside csm directory

from csm.generator import load_csm_1b, Segment
import time
import torchaudio
import torch

print("Hello from test_generate.py")
print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA device count:", torch.cuda.device_count())
print("Torch CUDA current device:", torch.cuda.current_device())

# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

device = 'cpu'
# device = 'cuda'

generator = load_csm_1b(device=device, model_path="models/csm-1b")

# audio = generator.generate(
#     text="Haha! hey there. Hello from Sesame. what's up?",
#     speaker=1,
#     context=[],
#     max_audio_length_ms=60_000,
# )


# torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

# -=--------
# load_csm_1b
speakers = [0, 1, 0, 1]
transcripts = [
    "Did you watch the i-show-speed's streaming last night?",
    "Yes! I watched the whole show! It's amazing!",
    "Right? That's fantastic! I mean the food.. the people.. their reaction... everything!",
    "Totally agreed. I wish he could do more this kind of travel and streaming.",
]
# should run under csm-test directory.
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
print("Starting to generate audio with context...")
start_time = time.time()

audio = generator.generate(
    # text="Me too, this is some cool stuff huh?",
    text="It's that classic tension between individual will, which is Mars, especially in fiery Leo... and the bigger power structures linked to Pluto, like the state and big corporations.",
    # text="Well, okay, hmm... haha, it's that classic tension between individual will, which is Mars, especially in fiery Leo... and the bigger power structures linked to Pluto, like the state and big corporations.",
    speaker=1,
    context=segments,
    max_audio_length_ms=60_000,
)
# print time spent in milliseconds precision
print(f"Time spent: {int((time.time() - start_time) * 1000)} ms")

torchaudio.save("audio_with_context_han.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

# warning: Clone succeeded, but checkout failed.iB/s
# You can inspect what was checked out with 'git status'
# and retry with 'git restore --source=HEAD :/'

# Usage: 
# cd csm-test
# cd D:\repos\chirp3_process\csm_process\csm-test
# python ..\generate-with-context.py
