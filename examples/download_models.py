import os
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from transformers import AutoTokenizer
import os

def download_and_save_tokenizer(save_dir: str = "models/csm-1b-tokenizer-saved"):
    """Download tokenizer and save locally"""
    try:
        print(f"Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "kyutai/moshiko-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors",
            trust_remote_code=True
        )
        
        # Save tokenizer
        print(f"Saving tokenizer to {save_dir}")
        tokenizer.save_pretrained(save_dir)
        
        # Verify the save
        print("Verifying saved tokenizer...")
        local_tokenizer = AutoTokenizer.from_pretrained(
            save_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        print("✓ Tokenizer saved and verified successfully!")
        
    except Exception as e:
        print(f"Error downloading/saving tokenizer: {e}")
        raise

    
def download_csm_model(
    model_id: str = "sesame/csm-1b",
    cache_dir: str = "models",
    token: str = None
):
    """Download model files to local directory"""
    if token is None:
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download model files
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=token,
        local_dir=os.path.join(cache_dir, "csm-1b"),
        local_dir_use_symlinks=False  # Save actual files, not symlinks
    )
    
    print(f"Model downloaded to: {local_dir}")
    return local_dir

if __name__ == "__main__":
    # Set your HuggingFace token if needed
    os.environ["HUGGING_FACE_HUB_TOKEN"] = "your_token_here"  # Optional
    
    download_and_save_tokenizer()
    print("Tokenizer download and save complete!")

    try:
        model_path = download_csm_model()
        print("Download complete! You can now use the model with:")
        print(f'generator = load_csm_1b(model_path="{model_path}")')
    except Exception as e:
        print(f"Download failed: {e}")