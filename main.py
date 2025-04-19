# main.py
import os
import logging
import asyncio
import pypdf
from io import BytesIO
import yaml
from typing import Dict
from datetime import datetime, timedelta
import threading

import google.generativeai as genai
from flask import Flask, jsonify, Request
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import httpx
import httpcore
from telegram.error import TimedOut, RetryAfter
import tenacity

# --- Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables (.env then .env.yaml fallback)
try:
    load_dotenv()
    try:
        with open('.env.yaml', 'r') as f:
            env_vars = yaml.safe_load(f)
            for k, v in env_vars.items():
                os.environ.setdefault(k, str(v))
    except FileNotFoundError:
        logger.info("No .env.yaml found, skipping")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing .env.yaml: {e}")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY")
ALLOWED_USER_IDS_STR = os.environ.get("ALLOWED_TELEGRAM_IDS", "")

try:
    ALLOWED_USER_IDS = {
        int(uid.strip())
        for uid in ALLOWED_USER_IDS_STR.split(',')
        if uid.strip()
    }
    logger.info(f"Allowed User IDs: {ALLOWED_USER_IDS}")
except ValueError as e:
    logger.error(f"Invalid ALLOWED_TELEGRAM_IDS: {e}")
    ALLOWED_USER_IDS = set()

# Configure Gemini
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model_text   = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        gemini_model_vision = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    except Exception as e:
        logger.error(f"Gemini init error: {e}")
        gemini_model_text = gemini_model_vision = None
else:
    gemini_model_text = gemini_model_vision = None
    logger.error("GEMINI_API_KEY not set")

# Limits & prompts
MAX_TEXT_LENGTH      = 4000
MAX_FILE_SIZE        = 20 * 1024 * 1024
MAX_PDF_TEXT_LENGTH  = 10000
MAX_IMAGE_SIZE       = 10 * 1024 * 1024
MAX_REQUESTS_PER_MIN  = 3
REQUEST_TIMEOUT      = 300
MAX_CONCURRENT_REQS  = 10

UNIFIED_PROMPT = """Please analyze this content and provide a structured response with the following sections:
1. Content Type: What type of content is this? (e.g., image, PDF, document)
2. Key Details: What are the main elements, text, or numbers visible? Limit to 3 lines
3. Context: What is the overall context or purpose?
4. Analysis: What are the key takeaways? (If medical, explain in layman's terms)

Please keep the response concise and easy to understand."""

# Thread‑safe rate/concurrency tracker
class RequestTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: Dict[int, datetime] = {}
        self._timestamps: Dict[int, list]    = {}
        self._concurrent: Dict[int, int]     = {}

    def is_rate_limited(self, user_id:int) -> bool:
        with self._lock:
            now = datetime.now()

            # prune old
            self._timestamps.setdefault(user_id, [])
            self._timestamps[user_id] = [
                ts for ts in self._timestamps[user_id]
                if now - ts < timedelta(minutes=1)
            ]

            if self._concurrent.get(user_id,0) >= MAX_CONCURRENT_REQS:
                return True

            if user_id in self._active:
                if now - self._active[user_id] < timedelta(seconds=REQUEST_TIMEOUT):
                    return True
                else:
                    del self._active[user_id]

            if len(self._timestamps[user_id]) >= MAX_REQUESTS_PER_MIN:
                return True

            return False

    def start(self, user_id:int):
        with self._lock:
            now = datetime.now()
            self._active[user_id] = now
            self._timestamps.setdefault(user_id, []).append(now)
            self._concurrent[user_id] = self._concurrent.get(user_id,0) + 1

    def done(self, user_id:int):
        with self._lock:
            self._active.pop(user_id, None)
            self._concurrent[user_id] = max(0, self._concurrent.get(user_id,0)-1)

request_tracker = RequestTracker()

# AI helper
async def get_ai_response(prompt:str, image_bytes:bytes|None=None) -> str:
    if not (gemini_model_text or gemini_model_vision):
        return "AI Service not configured."
    model = gemini_model_vision if image_bytes else gemini_model_text
    try:
        if image_bytes:
            image_part = {"mime_type":"image/jpeg","data":image_bytes}
            full_prompt = [prompt, image_part]
        else:
            full_prompt = prompt

        resp = await model.generate_content_async(full_prompt)
        if resp.parts:
            return resp.text
        return "AI response was empty or blocked."
    except Exception as e:
        logger.error(f"Gemini call error: {e}", exc_info=True)
        return f"Error contacting AI: {e}"

# Retry decorator for handling timeouts
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((TimedOut, ConnectionError, httpcore.ConnectTimeout)),
    before_sleep=lambda retry_state: logger.info(f"Retrying after error: {retry_state.outcome.exception()}")
)
async def send_message_with_retry(message, text):
    """Send a message with retry logic"""
    return await message.reply_text(text)

# Telegram handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    try:
        logger.info(f"Received /start command from user {update.effective_user.id}")
        uid = update.effective_user.id
        if uid not in ALLOWED_USER_IDS:
            logger.warning(f"Access denied for user {uid}")
            await send_message_with_retry(update.message, "Private bot. Access denied.")
            return
        logger.info(f"Access granted for user {uid}")
        await send_message_with_retry(update.message, "Hello! Send me text, a PDF or an image.")
    except Exception as e:
        logger.error(f"Error in start command: {str(e)}", exc_info=True)
        raise

async def handle_message(update, context):
    uid = update.effective_user.id
    cid = update.effective_chat.id

    if uid not in ALLOWED_USER_IDS:
        return

    if request_tracker.is_rate_limited(uid):
        await update.message.reply_text("Rate limit exceeded; please wait.")
        return

    request_tracker.start(uid)
    try:
        prompt = ""
        img_bytes = None

        msg = update.message
        if msg.text:
            if len(msg.text) > MAX_TEXT_LENGTH:
                await msg.reply_text(f"Text too long. Max {MAX_TEXT_LENGTH} chars.")
                return
            prompt = msg.text + "\n\n(Please be concise.)"

        elif msg.photo:
            photo = msg.photo[-1]
            if photo.file_size and photo.file_size > MAX_IMAGE_SIZE:
                await msg.reply_text("Image too large.")
                return
            await context.bot.send_chat_action(cid, "typing")
            file = await photo.get_file()
            bio = BytesIO()
            await file.download_to_memory(bio)
            img_bytes = bio.getvalue()
            prompt = msg.caption if msg.caption else UNIFIED_PROMPT
            if not msg.caption:
                await msg.reply_text("Analyzing the image...Please wait for a minute.")

        elif msg.document:
            doc = msg.document
            if doc.file_size and doc.file_size > MAX_FILE_SIZE:
                await msg.reply_text("File too large.")
                return
            await context.bot.send_chat_action(cid, "upload_document")
            file = await doc.get_file()
            bio = BytesIO()
            await file.download_to_memory(bio)
            data = bio.getvalue()

            if doc.mime_type == 'application/pdf':
                try:
                    reader = pypdf.PdfReader(BytesIO(data))
                    text = ""
                    for page in reader.pages:
                        text += (page.extract_text() or "") + "\n"
                        if len(text) > MAX_PDF_TEXT_LENGTH:
                            text = text[:MAX_PDF_TEXT_LENGTH]
                            break
                    
                    # If no text was extracted, treat it as an image-based PDF
                    if not text.strip():
                        logger.info("No text extracted from PDF, treating as image-based PDF")
                        img_bytes = data
                        prompt = msg.caption if msg.caption else UNIFIED_PROMPT
                        if not msg.caption:
                            await msg.reply_text("Analyzing the image-based PDF...Please wait for a minute.")
                    else:
                        prompt = (msg.caption if msg.caption else UNIFIED_PROMPT) + "\n\n" + text
                        if not msg.caption:
                            await msg.reply_text("Analyzing the PDF document...Please wait for a minute.")
                except Exception as e:
                    logger.error(f"PDF processing error: {e}", exc_info=True)
                    await msg.reply_text("Couldn't read this PDF. Please make sure it's not password protected and contains readable text.")
                    return
            elif doc.mime_type in ('image/jpeg','image/png'):
                if len(data) > MAX_IMAGE_SIZE:
                    await msg.reply_text("Image too large.")
                    return
                img_bytes = data
                prompt = msg.caption if msg.caption else UNIFIED_PROMPT
                if not msg.caption:
                    await msg.reply_text("Analyzing the image...Please wait for a minute.")
            else:
                await msg.reply_text(f"Unsupported file type: {doc.mime_type}. Please send a PDF or image file.")
                return

        else:
            await msg.reply_text("Send text, photo or PDF/image doc only.")
            return

        if len(prompt) > MAX_TEXT_LENGTH:
            prompt = prompt[:MAX_TEXT_LENGTH]

        await context.bot.send_chat_action(cid, "typing")
        ai_text = await get_ai_response(prompt, img_bytes)
        final = f"{ai_text}\n\n— Still unclear? Ask Ankit"

        # split into 4096‐char chunks
        for i in range(0, len(final), 4096):
            await context.bot.send_message(chat_id=cid, text=final[i:i+4096])

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        await context.bot.send_message(chat_id=cid, text="Processing error.")
    finally:
        request_tracker.done(uid)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    error = context.error
    
    # Log the error with full context
    logger.error(f"Exception while handling an update: {str(error)}", exc_info=error)
    logger.error(f"Update that caused error: {update}")
    if hasattr(context, 'chat_data'):
        logger.error(f"Chat data: {context.chat_data}")
    if hasattr(context, 'user_data'):
        logger.error(f"User data: {context.user_data}")
    
    # Handle specific error types with more detailed messages
    if isinstance(error, TimedOut):
        message = "Request timed out. Please try again."
    elif isinstance(error, (httpcore.ConnectTimeout, httpx.ConnectTimeout)):
        message = "Connection to Telegram timed out. Please try again in a moment."
    elif isinstance(error, RetryAfter):
        message = f"Too many requests. Please wait {error.retry_after} seconds."
    elif isinstance(error, ConnectionError):
        message = "Connection error. Please try again."
    elif isinstance(error, Exception):
        message = f"Error: {str(error)}"
    else:
        message = "Sorry, I encountered an error processing your request."
    
    # Try to send error message to user with retry
    try:
        if update and hasattr(update, 'effective_message') and update.effective_message:
            await send_message_with_retry(update.effective_message, message)
    except Exception as e:
        logger.error(f"Failed to send error message: {e}", exc_info=True)

# Build & initialize the bot (once)
application = None
bot_loop    = None

def create_application():
    global application, bot_loop
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return None

    logger.info("Creating Telegram bot application...")
    
    # Configure connection pool and timeouts
    try:
        # Use more aggressive timeouts for GCF environment
        application = (
            Application.builder()
            .token(TELEGRAM_BOT_TOKEN)
            .concurrent_updates(True)
            .connect_timeout(10.0)     # Reduced from 30 to be more aggressive
            .read_timeout(10.0)        # Reduced from 30 to be more aggressive
            .write_timeout(10.0)       # Reduced from 30 to be more aggressive
            .pool_timeout(10.0)        # Reduced from 30 to be more aggressive
            .connection_pool_size(8)   # Increase connection pool size
            .build()
        )
        logger.info("Application built successfully")

        # Register command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(
            MessageHandler(filters.TEXT | filters.PHOTO | (filters.Document.ALL & ~filters.COMMAND),
                          handle_message)
        )
        logger.info("Handlers registered successfully")
        
        # Register error handler
        application.add_error_handler(error_handler)
        logger.info("Error handler registered")

        # create & keep one loop alive with longer timeout
        logger.info("Creating event loop...")
        bot_loop = asyncio.new_event_loop()
        bot_loop.set_debug(True)  # Enable debug mode for event loop
        asyncio.set_event_loop(bot_loop)
        try:
            logger.info("Initializing application...")
            bot_loop.run_until_complete(application.initialize())
            logger.info("Bot initialized successfully on its own event loop")
            return application
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"Error creating application: {e}", exc_info=True)
        return None

# Create and verify application
application = create_application()
if application:
    logger.info("Bot application created and initialized successfully")
else:
    logger.error("Failed to create or initialize bot application")

# Cloud Functions Gen2 entrypoint
def app(request: Request):
    if request.method == "GET":
        return jsonify({
            "status": "ok",
            "bot_initialized": application is not None
        }), 200

    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    if application is None:
        return jsonify({"error": "Bot not initialized"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON"}), 400

    try:
        update = Update.de_json(data, application.bot)
    except Exception as e:
        logger.error(f"Invalid update JSON: {e}")
        return jsonify({"error": "Invalid update"}), 400

    # Drive the one shared loop
    try:
        bot_loop.run_until_complete(application.process_update(update))
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Error processing update: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Local testing via polling
if __name__ == "__main__":
    if TELEGRAM_BOT_TOKEN and GEMINI_API_KEY and ALLOWED_USER_IDS and application:
        logger.info("Starting local polling...")
        bot_loop.run_until_complete(application.start())
        application.run_polling(stop_signals=None)
    else:
        logger.error("Missing config—cannot start polling.")
