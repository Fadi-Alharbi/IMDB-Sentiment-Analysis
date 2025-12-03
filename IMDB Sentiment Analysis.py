import pandas as pd
import numpy as np
import pickle
import os  # Used to check if the file exists locally
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Settings ---
vocab_size = 10000
maxlen = 200
# Ensure this file is in the same folder as your Python script
filename = 'final data set1.xlsx'

# --- 2. Load data (VS Code Version) ---
print(f"🔍 Searching for file: {"final data set1.xlsx"} ...")

if os.path.exists(filename):
    try:
        df = pd.read_excel(filename)
        print(f"✅ Successfully read the file: {filename}")
    except Exception as e:
        print(f"❌ Error reading the file. Details: {e}")
        exit()
else:
    print(f"❌ Error: The file '{filename}' was not found.")
    print(f"📂 Current Working Directory: {os.getcwd()}")
    print("👉 Please make sure the Excel file is in the folder listed above.")
    exit()

# --- 3. Data Preprocessing ---
# Ensure necessary columns exist
if "Review" not in df.columns or "sentiment" not in df.columns:
    print("❌ Error: The Excel file must contain columns named 'Review' and 'sentiment'.")
    exit()

reviews = df["Review"].astype(str).fillna('').values

# Smart label cleaning function (handles '1' and 'positive')
def clean_label(x):
    # Convert to string, trim whitespace, and convert to lowercase
    s = str(x).strip().lower()

    # Rules:
    # If '1', '1.0', or 'positive' -> return 1 (Positive)
    if s == '1' or s == '1.0' or s == 'positive':
        return 1
    # Otherwise -> return 0 (Negative)
    return 0

# Apply the cleaning function
labels = df["sentiment"].apply(clean_label).values

# --- 🛑 Sanity Check ---
pos_count = np.sum(labels)
print(f"\n📊 Data Report:")
print(f"   - Number of positive reviews (1): {pos_count}")
print(f"   - Number of negative reviews (0): {len(labels) - pos_count}")

if pos_count == 0:
    print("⚠️ Warning: Positive count is 0! Check your data.")
else:
    print("✅ Great! Positive labels detected successfully.")

# --- 4. Tokenizer ---
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='pre')

# --- 5. Training ---
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(vocab_size, 32, input_length=maxlen),
    LSTM(64, dropout=0.3, recurrent_dropout=0.0), # recurrent_dropout=0 is optimized for standard hardware
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n🚀 Training started...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stop])

# --- 6. Save Model ---
model.save('final_model.keras')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n🎉 Saved! The model 'final_model.keras' and 'tokenizer.pickle' are ready.")