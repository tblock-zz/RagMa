import os
import argparse
# Disable telemetry first
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Monkey-patch posthog to fix ChromaDB's incorrect 3-argument call
try:
    import posthog
    original_capture = posthog.capture
    def patched_capture(*args, **kwargs):
        # ChromaDB 0.5.20 calls posthog.capture(user_id, event_name, properties)
        # But posthog-python expects (event_name, **kwargs) where distinct_id is a keyword arg.
        if len(args) == 3 and not kwargs:
            return original_capture(event=args[1], distinct_id=args[0], properties=args[2], **kwargs)
        return original_capture(*args, **kwargs)
    posthog.capture = patched_capture
except ImportError:
    pass

import llama_index
from dotenv import load_dotenv
from .ui import LocalChatbotUI
from .ui.theme import JS_LIGHT_THEME, CSS
import gradio as gr
from .pipeline import LocalRAGPipeline
from .logger import Logger
from .ollama import run_ollama_server, is_port_open

load_dotenv()

# CONSTANTS
LOG_FILE = "logging.log"
DATA_DIR = "data/data"
AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]

# PARSER
parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", help="Share gradio app")
args = parser.parse_args()

# OLLAMA SERVER
port_number = 11434
if not is_port_open(port_number):
    run_ollama_server()

# LOGGER

llama_index.core.set_global_handler("simple")
logger = Logger(LOG_FILE)
logger.reset_logs()

# PIPELINE
pipeline = LocalRAGPipeline()

# UI
ui = LocalChatbotUI(
    pipeline=pipeline,
    logger=logger,
    data_dir=DATA_DIR,
    avatar_images=AVATAR_IMAGES,
)

ui.build().launch(
    share=args.share, 
    server_name="0.0.0.0", 
    debug=False
)
