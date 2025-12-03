import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Settings ---
# ⚠️ CRITICAL: This number must match the 'maxlen' used in your training script.
# In the previous script, we set it to 200, so it must be 200 here as well.
maxlen = 200 

# File names (Must match the files saved by your training script)
model_filename = 'final_model.keras'
tokenizer_filename = 'tokenizer.pickle'

# --- 2. Load model and tokenizer ---
print("⏳ Loading model and tokenizer...")

# Check if the model and tokenizer files exist
if not os.path.exists(model_filename) or not os.path.exists(tokenizer_filename):
    print(f"❌ Error: The files '{model_filename}' or '{tokenizer_filename}' were not found!")
    print(f"📂 Current Folder: {os.getcwd()}")
    print("👉 Please run your training script first to generate these files.")
    exit()

# Load the trained model
model = load_model(model_filename)

# Load the tokenizer
with open(tokenizer_filename, 'rb') as handle:
    tokenizer = pickle.load(handle)

print("✅ System is ready! The model loaded successfully.")
print("--------------------------------------------------")

# --- 3. Interaction Loop ---
while True:
    # Get user input
    text = input("\n✍️  Type a movie review (in English) [or type 'exit' to quit]: ")
    
    # Exit condition
    if text.lower() == 'exit':
        print("👋 Goodbye!")
        break
    
    # Skip empty input
    if not text.strip():
        continue

    # 1. Convert text to sequences
    seq = tokenizer.texts_to_sequences([text])

    # 2. Apply Padding
    # Note: We used padding='pre' in training, so we MUST use 'pre' here.
    padded = pad_sequences(seq, maxlen=maxlen, padding='pre')

    # 3. Make Prediction
    prediction = model.predict(padded, verbose=0)[0][0]

    # 4. Output Result
    if prediction > 0.5:
        label = "😄 Positive"
        # Green color code for terminal output
        color = "\033[92m"
    else:
        label = "😠 Negative"
        # Red color code for terminal output
        color = "\033[91m"

    # Print the final result with color
    print(f"   Result: {color}{label}\033[0m")
