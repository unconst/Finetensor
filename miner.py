

import torch
import base64
import hashlib
import argparse
import bittensor as bt
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM

# Load the config.
parser = argparse.ArgumentParser(description='Miner')
parser.add_argument('--repo', type=str )
parser.add_argument('--netuid', type=str, default = 9 )
bt.subtensor.add_args(parser)
bt.wallet.add_args(parser)
bt.logging.add_args(parser)
config = bt.config( parser )

bt.logging.info("Starting validation of configuration parameters.")
if config.repo is None:
    bt.logging.error("Configuration error: 'repo' is required. e.g. microsoft/Phi-3-mini-4k-instruct")
    raise ValueError('repo is required')
if config.netuid is None:
    bt.logging.error("Configuration error: 'netuid' is required.")
    raise ValueError('netuid is required')
bt.logging.info("Configuration parameters validated successfully.")

# Load your wallet.
bt.logging.info("Loading wallet with provided configuration.")
wallet = bt.wallet(config=config)
bt.logging.info("Wallet loaded successfully.")

# Create the subtensor connection
bt.logging.info("Creating subtensor connection.")
subtensor = bt.subtensor(config=config)
bt.logging.info("Subtensor connection established.")

# Sync metagraph
metagraph = subtensor.metagraph(config.netuid)
if wallet.hotkey.ss58_address not in metagraph.hotkeys:
    bt.logging.error(f"Hotkey {wallet.hotkey.ss58_address} not found in metagraph. You need to register a miner on subnet {config.netuid}")
    raise ValueError(f"Hotkey {wallet.hotkey.ss58_address} not found in metagraph. You need to register a miner on subnet {config.netuid}")

# Load the model
model = AutoModelForCausalLM.from_pretrained(config.repo, trust_remote_code=True)
        
# Compute model hash
bt.logging.info("Computing model hash.")
model_hash = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()
string_hash = hashlib.sha256((model_hash + wallet.hotkey.ss58_address).encode())
final_hash = base64.b64encode(string_hash.digest()).decode("utf-8")
bt.logging.info(f"Model hash computed: {final_hash}")

# Create commit.
namespace = config.repo.split("/")[0]
name = config.repo.split("/")[1]
commit_message = f"{namespace}:{name}:none:{final_hash}"
bt.logging.info(f"Commit message created: {commit_message}")

# Commit the model to the chain.
bt.logging.info("Committing model to the chain.")
subtensor.commit(
    wallet=wallet,
    netuid=config.netuid,
    data=commit_message
)
bt.logging.success(f"Commited: {commit_message} to the chain under wallet: {wallet.hotkey.ss58_address} on subnet: {config.netuid}.")
