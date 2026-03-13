# warp/config.py
#
# Program-level configuration for WARP CORE.
# This file is listed in .gitignore — never commit it with a real token.
#
# To enable auto-sync to Hugging Face Hub:
#   1. Generate a token at https://huggingface.co/settings/tokens
#      (write access to the target dataset repo)
#   2. Paste it below
#   3. This file stays local — users never see or configure the token

HF_TOKEN = ''   # paste your token here

# Dataset repo where annotations are uploaded, e.g. 'your-username/sets-warp-data'
HF_DATASET_REPO = ''
