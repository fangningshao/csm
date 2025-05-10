# Prerequisite: pip install -e   inside csm directory
import argh
from csm.generator import load_csm_1b, load_csm_1b_local, Segment
import time
import torchaudio
import torch
import os
import glob

FS_PATH = 'D:\\repos\\chirp3_process\\csm_process\\csm-test\\laughters'
gold_transcripts = []
# for line in open(os.path.join(FS_PATH, 'utt9-13-withmark.txt'), 'r', encoding='utf-8').readlines():
for line in open(os.path.join(FS_PATH, 'utt9-13-laughter.txt'), 'r', encoding='utf-8').readlines():
    line = line.strip()
    if line:
        gold_transcripts.append(line)

print("gold transcripts:", gold_transcripts)

FS_AUDIO = glob.glob(os.path.join(FS_PATH, '*.wav'))
print("FS_AUDIO:", FS_AUDIO)

# GOLD_IDXES = [0,1,2,3,4]
# GOLD_IDXES = [3,4]
# GOLD_IDXES = [9, 10, 11, 12, 13]
GOLD_IDXES = [12]
# GOLD_IDXES = [1,2,3,4]
GOLD_IDXES = [i - 9 for i in GOLD_IDXES]  # Adjusting the index to start from 0

conf = ''.join([str(x + 4) for x in GOLD_IDXES])
gold_transcripts = [gold_transcripts[i] for i in GOLD_IDXES]
FS_AUDIO = [FS_AUDIO[i] for i in GOLD_IDXES]

print("Hello from test_generate.py")
print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA device count:", torch.cuda.device_count())
print("Torch CUDA current device:", torch.cuda.current_device())

# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "`cuda`"
# else:
#     device = "cpu"

# device = 'cpu'
device = 'cuda'

# generator = load_csm_1b_local(device=device, model_path="D:\\models\\csm-1b")
generator = load_csm_1b_local(device=device, model_path="D:\\models\\csm-1b-saved")


audio_paths = FS_AUDIO
transcripts = gold_transcripts
speakers = [0] * len(audio_paths)

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

input_texts = """
We're about to (laughter) dive into some really fascinating astrological stuff that Pam Gregory has been chatting about - (laughter) sounds good, right?
(laughter) Plus, it was stationary, just kind of hanging out around the full moon.
(laughter) Pretty interesting stuff!
(laughter) Burgers can be loaded up with jalapeños, salsa, and guacamole.
There's this highway ramp, oh, the bus ramp that takes you up twenty stories straight!
I (laughter) love those moments... you know, where your assumptions get flipped (laughter) upside down.
If you’re looking for fun things to do, (laughter) check out TripAdvisor or HappyCow for local tips and reviews, (laughter) especially if you're a foodie.
And those mild temperatures - not too hot, not too cold - make it (laughter) way more comfortable to work, right?
But, um, it’s not all sunshine and rainbows... there are definitely some downsides, like (laughter) dealing with misinformation or feeling that pressure to keep up online.
But then you head over to the U.S., and burgers get (laughter) super creative with all these signature sauces, different cheese blends, and even wild toppings like avocado or (laughter) fried eggs.

"""

# input_texts = """
# We're about to dive into some really fascinating astrological stuff that Pam Gregory has been chatting about - sounds good, right?
# Plus, it was stationary, just kind of hanging out around the full moon.
# Pretty interesting stuff!
# Burgers can be loaded up with jalapeños, salsa, and guacamole.
# """

def main(output_dir='.'):
    # Check if output directory exists, if not create it
    os.makedirs(output_dir, exist_ok=True)

    for idx, text in enumerate(input_texts.strip().split("\n")):
        print("Generating text:", text)
        audio = generator.generate(
            text=text,
            speaker=0,
            context=segments,
            max_audio_length_ms=60_000,
        )
        # print time spent in milliseconds precision
        print(f"Time spent: {int((time.time() - start_time) * 1000)} ms")

        torchaudio.save(
            os.path.join(output_dir, f"audio_with_context_{idx}_{conf}.wav"), 
            audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Saved audio_with_context_{idx}_{conf}.wav")


# Usage: 
# cd csm-test
# cd D:\repos\chirp3_process\csm_process\csm-test
# python ..\generate-with-context.py


if __name__ == "__main__":
    argh.dispatch_command(main)

# Usage:
# cd csm-test
# python ..\generate_laughters.py  -o 678   
# python ..\generate_laughters.py  -o 678_testcuda