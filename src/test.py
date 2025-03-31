import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pdfplumber
import os
import pickle
import gzip
from pathlib import Path

# Configuration
FOLDER_PATH = "../materials"
CACHE_PATH = "../cache"
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using AMD GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


def get_latest_pdf_mtime(FOLDER_PATH):
    """Get the latest modification time of PDF files in the folder"""
    pdf_files = [f for f in Path(FOLDER_PATH).glob('*.pdf')]
    if not pdf_files:
        raise ValueError("No PDF files found in the folder")
    return max(f.stat().st_mtime for f in pdf_files)


def load_or_extract_text(FOLDER_PATH, cache_path):
    """Load text from cache or extract from PDFs if cache is stale"""
    # Check if cache exists and is newer than all PDFs
    if Path(cache_path).exists():
        cache_mtime = Path(cache_path).stat().st_mtime
        pdf_mtime = get_latest_pdf_mtime(FOLDER_PATH)

        if cache_mtime > pdf_mtime:
            print("Loading cached text...")
            with gzip.open(cache_path, 'rb') as f:
                return pickle.load(f)

    # Extract text from PDFs if cache is invalid/missing
    print("Extracting text from PDFs...")
    text = ""
    for pdf_file in Path(FOLDER_PATH).glob('*.pdf'):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                print(f"Processing {pdf_file.name}...")
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

    # Save to compressed cache
    with open(cache_path, 'wb') as f:
        f.write(text)

    return text


# Load or extract text with caching
text = load_or_extract_text(FOLDER_PATH, CACHE_PATH)

# Rest of the code remains the same as previous version...
# [Character vocabulary creation, dataset preparation, model definition, training loop, etc.]

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode text to integers
encoded_text = [char_to_idx[ch] for ch in text]

# Create sequence datasets
sequences = []
for i in range(len(encoded_text) - SEQ_LENGTH):
    seq_in = encoded_text[i:i + SEQ_LENGTH]
    seq_out = encoded_text[i + 1:i + SEQ_LENGTH + 1]
    sequences.append((seq_in, seq_out))


# Dataset and DataLoader (remain same as before)
class CharDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_in, seq_out = self.sequences[idx]
        return torch.tensor(seq_in), torch.tensor(seq_out)


dataset = CharDataset(sequences)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model definition (remain same as before)
class CharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


model = CharLM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop (remain same as before)
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}')


# Generation function (remain same as before)
def generate_text(start_str, max_length=500, temperature=1.0):
    model.eval()
    generated = list(start_str)
    input_seq = torch.tensor([char_to_idx[ch] for ch in start_str[-SEQ_LENGTH:]], dtype=torch.long).unsqueeze(0).to(
        DEVICE)
    hidden = None

    for _ in range(max_length):
        outputs, hidden = model(input_seq, hidden)
        logits = outputs[0, -1, :] / temperature
        probabilities = torch.softmax(logits, dim=-1).cpu()
        next_char = torch.multinomial(probabilities, 1).item()
        generated.append(idx_to_char[next_char])
        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_char]], device=DEVICE)], dim=1)

    return ''.join(generated)


print(generate_text("The ", max_length=500))