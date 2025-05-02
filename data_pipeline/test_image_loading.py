# data_pipeline/test_image_loading.py
import json
import logging
import os
from llm_utils import LanguageModel # Assuming llm_utils.py is in the same directory or Python path

# --- Configuration ---
JSON_FILE_PATH = "/mnt/weka/aisg/ob1/MM-PRM/data_pipeline/formatted_data/1_formatted.json"
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for more verbose output
# --- End Configuration ---

# --- Logging Setup ---
# Configure root logger
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
# Configure logger for the 'main' namespace used in llm_utils
logger = logging.getLogger('main')
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
# --- End Logging Setup ---


def run_test():
    """Loads data, initializes the model, and runs generate_results."""
    logger.info(f"Starting test with JSON file: {JSON_FILE_PATH}")

    if not os.path.exists(JSON_FILE_PATH):
        logger.error(f"JSON file not found: {JSON_FILE_PATH}")
        return

    try:
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON data loaded successfully: {data}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return

    # Assuming the JSON contains a list of dictionaries
    if not isinstance(data, list) or not data:
        logger.error("JSON data is not a non-empty list.")
        return

    first_item = data[0]

    # --- Extract prompt and image path ---
    # Adjust these keys if they are different in your JSON structure
    question_text = first_item.get('question') # Load question from 'question' key
    image_path_from_json = first_item.get('image_path') # Renamed for clarity
    # ---

    if not question_text:
        logger.error("Could not find 'question' key in the first JSON item.")
        return
    if not image_path_from_json:
        logger.error("Could not find 'image_path' key in the first JSON item. Proceeding without image.")
        image_path = None # Explicitly set image_path to None
    else:
        # Resolve the image path: make it absolute relative to the JSON file's directory
        if os.path.isabs(image_path_from_json):
            image_path = image_path_from_json
            logger.info(f"Absolute image path found in JSON: {image_path}")
        else:
            abs_json_path = os.path.abspath(JSON_FILE_PATH)
            json_dir = os.path.dirname(abs_json_path)
            image_path = os.path.join(json_dir, image_path_from_json)
            image_path = os.path.normpath(image_path) # Clean up path (e.g., remove ../)
            logger.info(f"Relative image path '{image_path_from_json}' found in JSON. Resolved relative to JSON directory: {image_path}")

    # Now image_path holds the absolute path (or None)

    # Construct the prompt using the format from OmegaPRM AND llm_utils parsing needs
    prompt = (
        f"You are given an image. Your task is to answer the question based on the image provided. "
        f"Think step-by-step between <step> tags"
        f"Once you are done thinking, provide the final answer between <answer> tags, starting with 'The final answer is ..'.\n\n"
        f"Question: {question_text}\n"
    )

    # prompt = f"Your task is to answer the question below. Give complete reasoning steps before you answer, and when you are ready to answer, use Answer: The final answer is ..\n\nQuestion: {question_text}"

    logger.info(f"Resolved Image Path: {image_path}") # Updated log message

    if image_path and not os.path.exists(image_path):
         logger.error(f"Image file does not exist at resolved path: {image_path}") # Updated log message
         # Decide if you want to proceed without image or stop
         # return # Uncomment to stop if image doesn't exist

    # Format as file URI if it's a local path and exists
    formatted_image_path = None
    if image_path and os.path.exists(image_path):
        # Pass the absolute path directly, without 'file://' prefix
        formatted_image_path = image_path
        logger.info(f"Using absolute image path for model: {formatted_image_path}") # Updated log message
    elif image_path: # Path was provided/resolved but doesn't exist
         logger.warning(f"Resolved image path provided but file not found: {image_path}. Proceeding without image.") # Updated log message
    else: # No image path provided or found in JSON
        logger.info("No image path provided or found. Proceeding without image.")

    try:
        logger.info("Initializing LanguageModel...")
        # You might need to adjust model parameters if they differ from defaults
        lm = LanguageModel(
             model='Qwen/Qwen2.5-VL-32B-Instruct' # Ensure this matches the model used in run_omegaprm.sh
             # Add other parameters like temperature, top_k, top_p if needed
        )
        logger.info("LanguageModel initialized.")

        logger.info("Calling generate_results...")
        # Pass the formatted path (which might be None)
        results = lm.generate_results(prompt=prompt, image_path=formatted_image_path, num_copies=16) # Use num_copies=1 for testing
        logger.info("generate_results finished.")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"An error occurred during model initialization or generation: {e}", exc_info=True) # Log traceback

if __name__ == "__main__":
    run_test() 