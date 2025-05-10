# Prerequisite: pip install -e   inside csm directory
import argh
from csm.generator import load_csm_1b, load_csm_1b_local, Segment
import time
import torchaudio
import torch
import os
import glob

FS_PATH = 'csm-test/fur'
gold_transcripts = []
for line in open(os.path.join(FS_PATH, 'utt14-15-mew.txt'), 'r', encoding='utf-8').readlines():
    line = line.strip()
    if line:
        gold_transcripts.append(line)

print("gold transcripts:", gold_transcripts)

FS_AUDIO = glob.glob(os.path.join(FS_PATH, '*.wav'))
print("FS_AUDIO:", FS_AUDIO)

GOLD_IDXES = [14, 15]
# GOLD_IDXES = [1,2,3,4]
GOLD_IDXES = [i - 14 for i in GOLD_IDXES]  # Adjusting the index to start from 0

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

device = 'cpu'

# generator = load_csm_1b_local(device=device, model_path="models/csm-1b")
generator = load_csm_1b_local(device=device, model_path="models/csm-1b-saved")


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

# input_texts = """
# Happy birthday to you, happy birthday to you, hey human I love you, happy birth day to you.
# Jingle bells, jingle bells, jingle all the way.
# Oh... my love, my darling, I've hungered for your touch, a long, lonely time.
# Fish, i want fish, now. hold on to my fish.
# """

# input_texts = """
# Mew, mew, mew.
# Mi-ao!
# Purr, purr, meow!
# Happy birthday to you.
# Seafood, seafood, seafood all the way.
# Oh... my love, my darling, I've hungered for your touch.
# Fish, i want fish, now. hold on to my fish.
# Dinner! Fish! Chicken! I like it.
# Happy shower, hapy bath.
# """

# input_texts = """
# Miao, miau, mii-ao!
# 喵呜？喵呜！
# 呜喵，喵，喵？
# nyan, nyan, nyan!
# にゃにゃにゃゃ!
# にゃんにゃんにゃん!
# Mi-ao!
# Miii-ao~~
# """

input_texts = """
Mi-ao, m-au, u-au.
nyan!
Mi-ao!
nya, mew, miao, da da da.
"""


def main(output_dir='.'):
    # Check if output directory exists, if not create it
    os.makedirs(output_dir, exist_ok=True)

    for idx, text in enumerate(input_texts.strip().split("/n")):
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
# python ../generate-with-context.py


if __name__ == "__main__":
    argh.dispatch_command(main)

# Usage:
# cd csm-test
# python ../generate_fur.py  -o 678   
# python ../generate_fur.py  -o 678_testcuda