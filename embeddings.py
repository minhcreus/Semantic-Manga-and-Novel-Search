import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load your data
df = pd.read_csv("meta_manga_novel.csv")

# Load model and compute embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
texts = df['fulltext'].fillna("").tolist()
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Save embeddings
np.save("novel_embeddings.npy", embeddings)

# Create and save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "novel_index.faiss")
