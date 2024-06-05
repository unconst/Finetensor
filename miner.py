# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const
# Copyright © 2024 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import base64
import hashlib
import argparse
import traceback
import bittensor as bt
from transformers import AutoModelForCausalLM

def parse_arguments():
    """
    Parse command line arguments necessary for the miner script.
    This function sets up the necessary arguments for the mining operation and returns a configuration object.
    """
    parser = argparse.ArgumentParser(description='Miner')
    parser.add_argument('--repo', type=str, required=True, help='Huggingface repository containing the model.')
    parser.add_argument('--netuid', type=str, default=9, help='Network UID for the subnet.')
    bt.subtensor.add_args(parser)
    bt.wallet.add_args(parser)
    bt.logging.add_args(parser)
    return bt.config(parser)

def validate_configuration(config):
    """
    Validate the provided configuration parameters.
    Ensures that all required configurations are set and logs the validation process.
    """
    bt.logging.info("Starting validation of configuration parameters.")
    try:
        if not config.repo:
            bt.logging.error("Configuration error: 'repo' is required. e.g. microsoft/Phi-3-mini-4k-instruct")
            raise ValueError('repo is required')
        if not config.netuid:
            bt.logging.error("Configuration error: 'netuid' is required.")
            raise ValueError('netuid is required')
        bt.logging.info("Configuration parameters validated successfully.")
    except Exception as e:
        bt.logging.error(f"Validation failed: {str(e)}")
        traceback.print_exc()
        raise

def load_wallet(config):
    """
    Load the wallet using the provided configuration.
    This function initializes the wallet with the configuration and logs the process.
    """
    bt.logging.info("Loading wallet with provided configuration.")
    try:
        wallet = bt.wallet(config=config)
        bt.logging.info("Wallet loaded successfully.")
        return wallet
    except Exception as e:
        bt.logging.error(f"Failed to load wallet: {str(e)}")
        traceback.print_exc()
        raise

def establish_subtensor_connection(config):
    """
    Establish a connection to the subtensor using the provided configuration.
    Logs the process of creating and verifying the connection to the subtensor.
    """
    bt.logging.info("Creating subtensor connection.")
    try:
        subtensor = bt.subtensor(config=config)
        bt.logging.info("Subtensor connection established.")
        return subtensor
    except Exception as e:
        bt.logging.error(f"Failed to establish subtensor connection: {str(e)}")
        traceback.print_exc()
        raise

def sync_metagraph(subtensor, wallet, netuid):
    """
    Synchronize the metagraph and check if the wallet's hotkey is registered.
    This function ensures that the wallet's hotkey is part of the metagraph, critical for mining operations.
    """
    try:
        metagraph = subtensor.metagraph(netuid)
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            error_message = f"Hotkey {wallet.hotkey.ss58_address} not found in metagraph. You need to register a miner on subnet {netuid}"
            bt.logging.error(error_message)
            raise ValueError(error_message)
    except Exception as e:
        bt.logging.error(f"Failed to sync metagraph: {str(e)}")
        traceback.print_exc()
        raise

def load_model(repo):
    """
    Load the model from the specified repository.
    Attempts to load a pre-trained model from Huggingface and logs the process.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        bt.logging.info(f"Model loaded successfully from {repo}")
        return model
    except Exception as e:
        bt.logging.error(f"Failed to load model from {repo}: {str(e)}")
        traceback.print_exc()
        raise

def compute_model_hash(model, wallet):
    """
    Compute the hash of the model and concatenate it with the wallet's hotkey address.
    This function generates a unique hash for the model which is used in the blockchain commit.
    """
    bt.logging.info("Computing model hash.")
    try:
        model_hash = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()
        string_hash = hashlib.sha256((model_hash + wallet.hotkey.ss58_address).encode())
        final_hash = base64.b64encode(string_hash.digest()).decode("utf-8")
        bt.logging.info(f"Model hash computed: {final_hash}")
        return final_hash
    except Exception as e:
        bt.logging.error(f"Failed to compute model hash: {str(e)}")
        traceback.print_exc()
        raise

def create_commit_message(repo, final_hash):
    """
    Create a commit message with the namespace, name, and hash.
    This function formats a commit message that includes the model's repository and hash.
    """
    namespace = repo.split("/")[0]
    name = repo.split("/")[1]
    commit_message = f"{namespace}:{name}:none:{final_hash}"
    bt.logging.info(f"Commit message created: {commit_message}")
    return commit_message

def commit_model_to_chain(subtensor, wallet, netuid, commit_message):
    """
    Commit the model to the blockchain.
    This function handles the blockchain transaction to register the model's hash and logs the process.
    """
    bt.logging.info("Committing model to the chain.")
    try:
        subtensor.commit(
            wallet=wallet,
            netuid=netuid,
            data=commit_message
        )
        bt.logging.success(f"Committed: {commit_message} to the chain under wallet: {wallet.hotkey.ss58_address} on subnet: {netuid}.")
    except Exception as e:
        bt.logging.error(f"Failed to commit model to chain: {str(e)}")
        traceback.print_exc()
        raise

def main():
    """
    Main function to run the miner script.
    Orchestrates the mining process by calling other functions in sequence and handling exceptions.
    """
    try:
        # Parse command line arguments into a configuration object
        config = parse_arguments() 
        
        # Validate the parsed configuration
        validate_configuration(config)
        
        # Load the wallet using the configuration details 
        wallet = load_wallet(config)
        
        # Establish a connection to the subtensor network
        subtensor = establish_subtensor_connection(config)
        
        # Synchronize the metagraph with the subtensor network
        sync_metagraph(subtensor, wallet, config.netuid)
        
        # Load the model from the repository specified in the configuration
        model = load_model(config.repo)
        
        # Compute a unique hash for the model concatenated with the wallet address
        final_hash = compute_model_hash(model, wallet)
        
        # Create a commit message for the blockchain
        commit_message = create_commit_message(config.repo, final_hash)
        
        # Commit the model to the blockchain using the generated message
        commit_model_to_chain(subtensor, wallet, config.netuid, commit_message)
        
    except Exception as e:
        bt.logging.error(f"An error occurred in the main function: {str(e)}")  # Log an error message if an exception occurs
        traceback.print_exc()  # Print the stack trace of the exception

if __name__ == "__main__":
    main()  # Execute the main function if the script is run as the main module
