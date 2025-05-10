
# Create a conda env

conda create -n csm python=3.12


# Install Pytorch latest version

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## For 5090: (as of 2025/4)

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128


## FFmpeg Installation for windows
1. Download FFmpeg from https://github.com/BtbN/FFmpeg-Builds/releases
2. Extract to `D:\softwared\ffmpeg-master-latest-win64-gpl-shared`
3. Add to PATH using command prompt (Run as Administrator):

```cmd
setx /M PATH "%PATH%;D:\softwares\ffmpeg-master-latest-win64-gpl-shared\bin"
```

4. Close and reopen command prompt
5. Verify installation:

```cmd
ffmpeg -version
```


# Other prerequisites

pip install triton-windows
pip install protobuf

## Now get the tokenizer

get it from https://huggingface.co/taylorj94/Llama-3.2-1B/tree/main/original

cd csm
pip install -e .  # install pkg to resolve deps
pip install -r requirements.txt
