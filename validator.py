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

# Standard library imports
import os
import json
import math
import random
import asyncio
import argparse
import traceback

# Third-party imports
import torch
import aiohttp
import bittensor as bt
from tqdm import tqdm
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from huggingface_hub import HfApi
from typing import List, Dict, Any, Optional, Tuple
from transformers import GPT2TokenizerFast, AutoModelForCausalLM

# Global constants which define incentive landscape.
EPSILON = 0.01  # Epsilon value for numerical stability
MAX_MODEL_SIZE = 1e7  # Maximum model size in bytes
QUESTIONS_PER_EVAL = 10  # Number of questions per evaluation
TOKENIZER_NAME = 'Xenova/gpt-4' # Tokenizer model name

# NOTE: Function to load and parse the validator configuration from command line arguments.
def load_validator_config() -> argparse.Namespace:
    """
    Load and parse the validator configuration from command line arguments.

    Returns:
        argparse.Namespace: Parsed configuration object.
    """
    bt.logging.debug("Initializing argument parser for validator configuration.")
    # Initialize the argument parser
    parser = argparse.ArgumentParser()

    # Add command line arguments with their types and default values
    parser.add_argument('--netuid', type=int, default=9, help="Network UID")
    parser.add_argument('--device', type=str, default= 'cuda:1' if torch.cuda.is_available() else 'cpu', help="Device to run the model on")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse the arguments and return the configuration
    config = bt.config(parser)
    bt.logging.success("Validator configuration loaded and parsed successfully.")
    return config

# Load the validator config
config = load_validator_config()
bt.logging.info(f"Loaded configuration: {config}")

# Check the user specified device.
if config.device == 'cpu':
    raise ValueError("CPU is not supported for this task. You must have a GPU with at least " + str(MAX_MODEL_SIZE / 1e9) + " GB of RAM.")
# GPUs must have at least 40GB of RAM to fit the MAX_MODEL_SIZE.
if 'cuda' in config.device:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. You must have a GPU with at least " + str(MAX_MODEL_SIZE / 1e9) + " GB of RAM.")
    if torch.cuda.get_device_properties(0).total_memory < MAX_MODEL_SIZE:
        raise ValueError("Insufficient GPU memory. You must have a GPU with at least " + str(MAX_MODEL_SIZE / 1e9) + " GB of RAM.")

# Create OpenAI client from API key
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    bt.logging.error("OPENAI_API_KEY environment variable is not set.")
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
CLIENT = AsyncOpenAI(api_key=api_key)
bt.logging.debug("OpenAI client created.")

# Create the global tokenizer.
try:
    TOKENIZER = GPT2TokenizerFast.from_pretrained( TOKENIZER_NAME )
    TOKENIZER.pad_token = TOKENIZER.eos_token
    bt.logging.success("Tokenizer loaded and configured.")
except Exception as e:
    bt.logging.error(f"Failed to load tokenizer: {str(e)}")
    raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

# Create the global subtensor instance.
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR is None:
        try:
            SUBTENSOR = bt.subtensor( config=config )
        except Exception as e:
            bt.logging.error(f"Failed to create subtensor connection: {str(e)}")
            await asyncio.sleep(5)  # Wait for 5 seconds before retrying
            return await get_subtensor(config)
    return SUBTENSOR

# NOTE: Function to fetch a random Wikipedia article title using the Wikipedia API.
async def get_random_wikipedia_article() -> str:
    """
    Fetches a random Wikipedia article title using the Wikipedia API.
    
    This function asynchronously contacts the Wikipedia API to retrieve the title
    of a random article. It uses the 'random' list feature of the API to get one
    random article title from the main namespace (namespace 0).
    
    Returns:
        str: The title of a randomly selected Wikipedia article.
    """
    bt.logging.trace("Preparing to fetch a random Wikipedia article.")
    # Wikipedia API endpoint for queries
    url = 'https://en.wikipedia.org/w/api.php'
    
    # Parameters for fetching a random article title
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'random',
        'rnnamespace': 0,  # Main namespace
        'rnlimit': 1       # Limit to one random article
    }
    
    # Create an asynchronous HTTP session
    async with aiohttp.ClientSession() as session:
        # Send a GET request to the Wikipedia API
        async with session.get(url, params=params) as response:
            # Parse the JSON response asynchronously
            data = await response.json()
            # Extract the article title from the response data
            title = data['query']['random'][0]['title']
    
    bt.logging.info(f"Random Wikipedia article fetched: {title}")
    return title

# NOTE: Function to asynchronously fetch and extract text sections from a Wikipedia article.
async def get_paragraphs_from_article(title: str) -> List[str]:
    """
    Asynchronously fetches and extracts text sections from a Wikipedia article.
    
    This function uses the Wikipedia API to retrieve the HTML content of a specified article.
    It then parses the HTML to extract text from paragraph and header tags.
    
    Args:
        title (str): The title of the Wikipedia article to fetch.
    
    Returns:
        List[str]: A list of text sections extracted from the article.
    """
    bt.logging.debug(f"Fetching content for Wikipedia article: {title}")
    # Wikipedia API endpoint and parameters for fetching article content
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'parse',
        'page': title,
        'format': 'json',
        'prop': 'text'
    }
    
    # Create an asynchronous HTTP session and fetch the article content
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            html_content = data['parse']['text']['*']
    
    # Parse the HTML content using BeautifulSoup to extract text sections
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = [section.get_text() for section in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
    
    bt.logging.info(f"Extracted {len(sections)} sections from article '{title}'.")
    return sections

# NOTE: Function to asynchronously retrieve a random paragraph from a random Wikipedia article.
async def get_random_paragraph() -> str:
    """
    Asynchronously retrieves a random paragraph from a random Wikipedia article.
    
    This function continuously fetches random Wikipedia articles until it finds one
    with a paragraph longer than 50 characters, then returns a random paragraph from
    those long paragraphs.
    
    Returns:
        str: A randomly selected paragraph longer than 50 characters.
    """
    bt.logging.trace("Starting to fetch a random paragraph from Wikipedia.")
    while True:
        title = await get_random_wikipedia_article()
        paragraphs = await get_paragraphs_from_article(title)
        # Filter paragraphs to find those longer than 50 characters
        long_paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 50]
        
        if long_paragraphs:
            # Select a random paragraph from the list of long paragraphs
            random_paragraph = random.choice(long_paragraphs)
            bt.logging.success(f"Random paragraph selected: {random_paragraph[:30]}...")  # Show a snippet of the paragraph
            return random_paragraph
        # If no long paragraphs are found, continue to fetch another article
        else:
            bt.logging.warning("No suitable paragraphs found, fetching another article.")
            continue
        

# NOTE: Function to asynchronously call the OpenAI API to generate completions based on the provided parameters.
async def call_openai(
    messages: List[Dict[str, Any]], 
    temperature: float, 
    model: str, 
    seed: Optional[int] = 1234, 
    response_format: Optional[Dict[str, Any]] = None, 
    top_p: Optional[float] = None
) -> str:
    """
    Asynchronously calls the OpenAI API to generate completions based on the provided parameters.
    
    Args:
        messages (List[Dict[str, Any]]): A list of message dictionaries, each containing 'role' and 'content'.
        temperature (float): Controls randomness in the generation. Lower is more deterministic.
        model (str): Specifies the model to be used for generating completions.
        seed (Optional[int]): An optional seed to ensure determinism, defaults to 1234.
        response_format (Optional[Dict[str, Any]]): Optional formatting options for the response.
        top_p (Optional[float]): An optional float specifying the nucleus sampling parameter.
    
    Returns:
        str: The content of the message from the first choice of the generated completions.
    """
    bt.logging.debug("Preparing to call OpenAI API for text generation.")
    response = await CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        seed=seed,
        response_format=response_format,
        top_p=top_p,
    )
    response_content = response.choices[0].message.content
    bt.logging.info("Received response from OpenAI API.")
    return response_content

# NOTE: Function to asynchronously generate a set of benchmark questions based on a random paragraph from Wikipedia.
async def generate_questions() -> List[Dict[str, Any]]:
    """
    Asynchronously generates a set of benchmark questions based on a random paragraph from Wikipedia.
    
    This function first retrieves a random paragraph, then constructs a prompt to generate questions
    using the OpenAI API. The generated questions are then shuffled to randomize the order of choices
    while preserving the correct answer.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing a question, shuffled choices, and the correct answer.
    """
    bt.logging.trace("Starting to generate benchmark questions.")
    # Fetch a random paragraph from Wikipedia
    paragraph = await get_random_paragraph()
    
    # Construct the prompt for the OpenAI API call
    # This prompt is designed to generate a synthetic benchmark by using a random Wikipedia paragraph as context.
    # The goal is to extract information from this context and transform it into a set of structured benchmark questions.
    # Each question will be formatted in JSON with fields for the question text, multiple choices, and the correct answer.
    # An example is provided within the prompt to guide the model on the expected output format and content.
    prompt = f"""
        Here is the context text: 
        context: {paragraph}
        
        Generate a set of 5 benchmark questions in JSON format regarding this context which has form:

        question: 
        choices:
        answer: 

        here is an example benchmark question:

        question: "Where would you find a seafood restaurant in the east coast of North America?"
        choices: {{"label": ["A", "B", "C", "D", "E"], "text": ["maine", "boston", "beach town", "coastal cities", "ocean"]}}
        answer: "A"
    """
    
    # Prepare the message for the API call
    messages = [{"role": "user", "content": prompt}]
    
    # Call the OpenAI API to generate questions
    questions = json.loads(await call_openai(
        messages=messages,
        temperature=0.2,
        model='gpt-4o',
        seed=None,  # TODO: Consider setting a seed for reproducibility
        response_format={"type": "json_object"},
    ))
    
    bt.logging.success("Questions generated successfully.")
    
    # Shuffle the choices in the generated questions to prevent answer bias
    shuffled_questions = []
    for quest in questions['questions']:
        # Find the index of the correct answer
        correct_answer_idx = quest['choices']['label'].index(quest['answer'])
        correct_answer = quest['choices']['text'][correct_answer_idx]
        
        # Shuffle the choices text
        new_choices_text = quest['choices']['text'][:]
        random.shuffle(new_choices_text)
        
        # Find the new label for the correct answer after shuffling
        new_answer_label = quest['choices']['label'][new_choices_text.index(correct_answer)]
        
        # Append the shuffled question to the list
        shuffled_questions.append({
            'question': quest['question'],
            'choices': {'label': quest['choices']['label'], 'text': new_choices_text},
            'answer': new_answer_label
        })
    
    bt.logging.info("Questions shuffled and prepared for evaluation.")
    return shuffled_questions

# NOTE: Function to fetch multiple sets of questions for evaluation.
async def get_questions( n_queries: int ):
    bt.logging.debug(f"Fetching {n_queries} sets of questions.")
    return sum(await asyncio.gather(*(generate_questions() for _ in range(n_queries))), [])


# NOTE: Function to asynchronously download a miner model based on a given hotkey from the blockchain metadata.
async def download_miner_model(hotkey: str) -> Tuple[Optional[int], Optional[torch.nn.Module]]:
    """
    Asynchronously downloads a miner model based on a given hotkey from the blockchain metadata.

    This function retrieves the model metadata from the blockchain using the hotkey, decodes the
    model repository ID from the metadata, and then downloads the model from the Hugging Face repository
    if it meets the size constraints specified in the configuration.

    Args:
        hotkey (str): The hotkey associated with the model to be downloaded.

    Returns:
        Tuple[Optional[int], Optional[torch.nn.Module]]: A tuple containing the block number and the
        downloaded model if successful, or (None, None) if an error occurs or the model is too large.
    """
    bt.logging.debug(f"Attempting to download miner model for hotkey: {hotkey}")
    try:        
        # Establish a connection to the subtensor network to access blockchain data.
        subtensor = await get_subtensor()
        
        # Retrieve the metadata associated with the given hotkey and network UID from the blockchain.
        metadata = bt.extrinsics.serving.get_metadata(subtensor, config.netuid, hotkey)
        
        # Extract the commitment data from the metadata which includes the encoded model repository ID.
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]  # Skip the '0x' prefix.
        block = metadata["block"]  # The blockchain number where this metadata was recorded.
        
        # Convert the hexadecimal data into a string to parse the model repository ID.
        chain_str = bytes.fromhex(hex_data).decode()
        model_repo_id = chain_str.split(':')[0] + '/' + chain_str.split(':')[1]  # Format: 'username/repository'
        
        # Create an instance of the Hugging Face API client to interact with the Hugging Face Hub.
        api = HfApi()
        
        # Fetch detailed information about the model from the Hugging Face repository using the repository ID.
        model_info = api.model_info(repo_id=model_repo_id, timeout=10, files_metadata=True)
        
        # Sum up the file sizes of the model to determine the total size.
        size = sum(repo_file.size for repo_file in model_info.siblings)
        
        # Verify if the total size of the model is within the permissible limit.
        if size > MAX_MODEL_SIZE:
            bt.logging.warning(f"Model too large to download: {size} bytes")
            raise ValueError(f"Model too large to download: {size} bytes")
        
        # If the size is within limits, proceed to download the model.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_repo_id,
            use_safetensors=True,  # Use safe tensors to ensure compatibility and safety.
        )
        
        bt.logging.success(f"Model downloaded successfully from {model_repo_id}.")
        # Return the block number and the downloaded model.
        return block, model
    
    except Exception as e:
        bt.logging.error(f"Failed to download model: {str(e)}")
        bt.logging.debug(f"Exception details: {traceback.format_exc()}")
        # Return None values indicating failure to download the model.
        return None, None
    
# NOTE: Function to compute the accuracy of a model on a given set of multiple-choice questions.
async def compute_accuracy(model: torch.nn.Module, questions: List[Dict[str, Any]]) -> Tuple[float, List[str], List[str]]:
    """
    Computes the accuracy of the model on a given set of multiple-choice questions.

    This function evaluates the model's ability to correctly answer multiple-choice questions.
    It formats the questions into a prompt, feeds them to the model, and interprets the logits
    to determine the most likely answer according to the model. It then compares these answers
    to the correct answers to calculate the overall accuracy.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        questions (List[Dict[str, Any]]): A list of dictionaries, each containing a question and its possible answers.

    Returns:
        Tuple[float, List[str], List[str]]: A tuple containing the accuracy percentage,
                                            a list of the model's guesses, and a list of the correct answers.
    """
    bt.logging.info("Starting model evaluation for accuracy computation.")
    # Ensure the model is in evaluation mode and moved to the correct device
    model.eval()  # Sets the model to evaluation mode, which turns off specific layers like dropout.
    model.to(config.device)  # Moves the model to the specified device (CPU or GPU), for efficient computation.
    
    n_correct = 0  # Counter for correct answers
    guesses = []  # List to store the model's guesses
    answers = []  # List to store the correct answers
    
    for question in questions:
        # Format the choices into a string for the prompt
        formatted_choices = ""
        for ans, lab in zip(question['choices']['text'], question['choices']['label']):
            formatted_choices += f"({lab}) {ans} "  # Formats each choice as "(Label) Choice"
        
        # Construct the prompt with additional context questions for better model performance
        # Explanation of Prompting Strategy:
        # This prompt is structured in a "two-shot" learning format, where two examples are provided before the actual question
        # that needs to be answered by the model. The "two-shot" method involves giving the model two examples of the task at hand,
        # which helps the model understand the context and the type of response expected. In this case, the task is to answer multiple-choice
        # questions correctly. By providing correct answers to similar questions, we prime the model to better understand and generate
        # the correct answer for the subsequent question. This approach can significantly enhance the model's performance by setting
        # a clear example of the task's requirements.
        prompt = f"""
        The following are multiple choice questions.
        
        What is one of the responsibilities of the Chief Minister of the ACT?
        (A) Overseeing the judiciary (B) Conducting federal elections (C) Managing foreign relations (D) Drafting national laws (E) Allocating executive power to ministers
        Answer: E
        
        In which Australian territory does the Chief Minister have the power to establish government 'administrative units'?
        (A) New South Wales (B) Queensland (C) Victoria (D) Australian Capital Territory (E) Western Australia
        Answer: D
                
        {question['question']}
        {formatted_choices}
        Answer: 
        """
        
        # Tokenize the prompt and move tensors to the correct device
        inputs = TOKENIZER(prompt, return_tensors='pt')  # Converts the prompt into a format the model can process.
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Ensures all input tensors are on the correct device.
        
        # Initialize list to store scores for each label
        label_scores = []
        
        # Perform inference without gradient calculation
        with torch.no_grad():  # Disables gradient calculation to save memory and computations, which is essential during inference.
            outputs = model(**inputs, labels=inputs['input_ids'])  # Feeds the tokenized prompt to the model and retrieves the outputs.
            max_score = -float('inf')  # Initialize max_score to a very low number to ensure any real score will be higher.
            max_label = None  # Initialize max_label to None, to be updated with the label of the highest score.
            
            # Calculate scores for each possible label and find the label with the maximum score
            for label in question['choices']['label']:
                # Extract the logits for the current label. Logits are the raw output scores from the model's last layer before any activation function like softmax is applied.
                # These scores are used to determine the model's confidence in each possible answer choice.
                # Here, we access the logits specifically for the current label. The `outputs.logits` tensor has dimensions where
                # the last dimension corresponds to the vocabulary indices. We use `TOKENIZER.encode(label)[1]` to find the vocabulary index of the current label.
                # The `[:, -1, index]` accesses the logit for the current label at the last position of the sequence from the batch (assuming a batch size of 1 here).
                label_score = outputs.logits[:, -1, TOKENIZER.encode(label)[1]].item()  # Convert the tensor to a Python float.
                label_scores.append(label_score)  # Append the extracted score to the list `label_scores` for later analysis or debugging.
                
                # Update max_score and max_label if the current label's score is higher than the max_score found so far.
                if label_score > max_score:
                    max_score = label_score
                    max_label = label  # Set max_label to the current label as it has the highest score so far.
            
            # Check if the predicted label matches the correct answer
            if max_label == question['answer']:
                n_correct += 1  # Increment the correct answer counter.
            
            guesses.append(max_label)  # Append the guessed label to the guesses list.
            answers.append(question['answer'])  # Append the correct answer to the answers list.
    
    # Calculate and return the accuracy, along with the guesses and correct answers
    accuracy = n_correct / len(questions) if questions else 0  # Calculate accuracy as the ratio of correct answers.
    bt.logging.success(f"Model evaluation completed. Accuracy: {accuracy*100:.2f}%")
    return accuracy, guesses, answers
    
# NOTE: Main asynchronous function to evaluate models on questions and update weights on the blockchain.
async def main():
    """
    Main asynchronous function to evaluate models on questions and update weights on the blockchain.
    
    This function initializes a validator wallet, continuously fetches questions, and evaluates multiple models
    fetched from the blockchain based on their accuracy on these questions. It then updates the weights on the
    blockchain for the model with the highest adjusted accuracy.
    """
    bt.logging.info("Starting main evaluation loop.")
    # Initialize the validator wallet using the configuration settings.
    wallet = bt.wallet(config=config)
    
    while True:
        try:
            # Fetch the subtensor connection.
            subtensor = await get_subtensor()

            # Fetch a set of questions for evaluation.
            questions = await get_questions(QUESTIONS_PER_EVAL)
            
            # Retrieve the current metagraph from the blockchain using the network UID.
            metagraph = subtensor.metagraph(config.netuid)
            
            # Initialize variables to track the highest accuracy and corresponding model details.
            highest_accuracy_so_far = 0.0
            highest_accuracy_uid = None
            highest_accuracy_block = math.inf
            # Iterate over each unique identifier (UID) in the metagraph to evaluate associated models.
            for uid in tqdm(metagraph.uids.tolist()):
                
                try:
                    # Asynchronously download the model using the UID's associated hotkey.
                    model_block, model = await download_miner_model(metagraph.hotkeys[uid])
                    
                    # If the model or its block number is not available, skip the evaluation for this model.
                    if model is None or model_block is None:
                        bt.logging.warning(f"Skipping model with UID {uid} due to download issues.")
                        continue
                    
                    # Compute the accuracy of the model on the fetched questions, which measures how well the model performs.
                    accuracy, _, _ = await compute_accuracy(model, questions)
                        
                    # Adjust the accuracy based on the block number to prioritize older or newer models.
                    # This adjustment helps to prevent model stealing by giving an advantage to earlier models.
                    if model_block < highest_accuracy_block:
                        # Boost accuracy for older models by a factor defined by `EPSILON`.
                        # `config.epsilon` is a small positive value that provides a scoring advantage to older models.
                        if accuracy * (1 + EPSILON) > highest_accuracy_so_far:
                            highest_accuracy_block = model_block
                            highest_accuracy_so_far = accuracy
                            highest_accuracy_uid = uid
                            bt.logging.info(f"New highest accuracy model found: UID {uid} with adjusted accuracy {accuracy*100:.2f}%.")
                            
                    elif model_block > highest_accuracy_block:
                        # Penalize newer models by comparing their raw accuracy against the boosted accuracy of the current best.
                        # This penalization discourages copying and slightly favors older models.
                        if accuracy > highest_accuracy_so_far * (1 + EPSILON):
                            highest_accuracy_block = model_block
                            highest_accuracy_so_far = accuracy
                            highest_accuracy_uid = uid
                            bt.logging.info(f"New highest accuracy model found: UID {uid} with adjusted accuracy {accuracy*100:.2f}%.")
                    else:
                        # For models from the same block, choose the one with the highest raw accuracy.
                        # This ensures that among contemporaneous models, the best performer is chosen.
                        if accuracy > highest_accuracy_so_far:
                            highest_accuracy_block = model_block
                            highest_accuracy_so_far = accuracy
                            highest_accuracy_uid = uid
                            bt.logging.info(f"New highest accuracy model found: UID {uid} with raw accuracy {accuracy*100:.2f}%.")
                            
                except Exception as e:
                    traceback.print_exc()
                    bt.logging.error(f"Error during model evaluation for UID {uid}: {str(e)}")
                    continue
            
            # Update the weights on the blockchain for the model with the highest adjusted accuracy.
            if highest_accuracy_uid is not None:
                bt.logging.info(f"Updating blockchain weights for model with UID {highest_accuracy_uid}.")
                weights = torch.zeros_like(metagraph.uids)
                weights[highest_accuracy_uid] = 1.0        
                subtensor.set_weights(
                    netuid=config.netuid,
                    wallet=wallet,
                    uids=metagraph.uids.tolist(),
                    weights=weights.tolist()
                )
                bt.logging.success("Blockchain weights updated successfully.")
            
        # Catch errors.
        except Exception as e:
            traceback.print_exc()
            bt.logging.error(f"Error during main evaluation loop: {str(e)}")
            await asyncio.sleep(10)  # Wait for 10 seconds before retrying the loop
            continue

if __name__ == "__main__":
    asyncio.run(main())
