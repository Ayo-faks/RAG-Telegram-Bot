import pandas as pd
import os
from dotenv import load_dotenv
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
DOMAIN = "developer.mozilla.org"

def remove_newlines(series):
    series = series.str.replace('\n', ' ')
    series = series.str.replace('\\n', ' ')
    series = series.str.replace('  ', ' ')
    series = series.str.replace('  ', ' ')
    return series

# Create a list to store the text files
texts = []

# Get all the text files in the specified directory
for file in os.listdir("text/" + DOMAIN + "/"):
    with open("text/" + DOMAIN + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()
        filename = file[:-4].replace('_', '/')

        if filename.endswith(".txt") or 'users/fxa/login' in filename:
            continue

        texts.append((filename, text))

# Create a Pandas DataFrame from the collected text
df = pd.DataFrame(texts, columns=['fname', 'text'])

# Clean and preprocess the 'text' column
df['text'] = df.fname + ". " + remove_newlines(df.text)

# Save the DataFrame to a CSV file
df.to_csv('processed/scraped.csv')

# Load a tokenizer for text processing
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")

# Load the processed data from the CSV file
df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Split text into smaller chunks that fit within the maximum sequence length (e.g., 512 tokens)
max_sequence_length = 512
df['chunks'] = df.text.apply(lambda x: [x[i:i+max_sequence_length] for i in range(0, len(x), max_sequence_length)])

# Initialize an empty list to store the embeddings
embeddings_list = []

# Initialize the model
model = AutoModel.from_pretrained("thenlper/gte-base")

# Process each batch of text and store the embeddings
batch_size = 1  # Adjust the batch size based on your resources
for index, row in df.iterrows():
    chunks = row['chunks']
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_sequence_length)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # You can use a pooling strategy here
        embeddings_list.append(embeddings)

# Concatenate the embeddings and convert to a tensor
embeddings_tensor = torch.cat(embeddings_list, dim=0)

# (Optionally) normalize embeddings
normalized_embeddings = normalize(embeddings_tensor, p=2, dim=1)

# Save the final DataFrame with text embeddings to a CSV file
df['embeddings'] = normalized_embeddings.tolist()
df.to_csv('processed/embeddings.csv')
