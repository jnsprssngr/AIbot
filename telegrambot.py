from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters
from transformers import pipeline

# Load model pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="bfloat16",
    device_map="auto"
)

# Function to generate responses
def generate_response(user_input: str) -> str:
     """Generate a response from the LLM model."""
     try:
         prompt = user_input

         # Generate a response
         outputs = pipe(
             prompt,
             max_new_tokens=150,
             do_sample=True,
             temperature=0.7,
             top_k=50,
             top_p=0.9
         )
         response = outputs[0]["generated_text"]

         # Remove echoed input if present
         if response.startswith(user_input):
             response = response[len(user_input):].strip()

         return response

     except Exception as e:
        print(f"Error during text generation: {e}")
        return "Sorry, something went wrong while generating the response."

# Define the /start command handler
async def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text("Hello! I am your AI assistant. How can I help you?")

# Define a handler for processing text messages
async def process_message(update: Update, context: CallbackContext) -> None:
    """Use the LLM to respond to user messages."""
    print(f"Received message: {update.message.text}")  # Debug log
    user_message = update.message.text
    response = generate_response(user_message)
    print(f"Generated response: {response}")  # Debug log
    await update.message.reply_text(response)

# Main function to initialize and run the bot
def main() -> None:

    """Start the bot."""
    API_TOKEN = "MypersonalandSecureToken"

    # Create the application
    application = Application.builder().token(API_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
