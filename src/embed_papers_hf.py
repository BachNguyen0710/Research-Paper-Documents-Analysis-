# simple script for embedding papers using huggingface Specter
# requirement: pip install torch transformers==4.2.2 pandas tqdm

from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib
import torch
import re
import pandas as pd


def remove_custom_stopwords(text, stopwords):
    words = text.split()
    filtered = [w for w in words if w.lower() not in stopwords]
    return ' '.join(filtered)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

custom_stopwords = {
    # stopwords cơ bản
    "the","and","a","an","of","to","in","for","on","with","by",
    "is","are","was","were","be","as","at","from","that","this",
    "it","or","which","can","have","has","had","but","not","also",
    "such","however","more","their","these","our","using","used",
    
    # Tự bỏ comment chạy nếu muốn loại bỏ stopword nào đó
    # stopwords thêm vào
    "research","study","studies",
    "data",
    # "analysis",
    "results",
    "based","between",
    "science",
    # "scientific",
    "students",
    # "education",
    "learning",
    "knowledge",
    "information",
    "most",
    "they", "all", "been", "how", "than", "authors", "article",
    "may", "one", "two", "new", "some", "other", 

    # Từ đặt biệt
    # "erroneously"
}

def preprocess_text(text):
    text = normalize_text(text)
    text = remove_custom_stopwords(text, custom_stopwords)
    return text

class Dataset:
    def __init__(self, data_path, max_length=512, batch_size=32, clean_stopword=False):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.max_length = max_length
        self.batch_size = batch_size

        df = pd.read_csv(data_path)
        df['title_clean'] = df['title'].apply(preprocess_text)
        df['abstract_clean'] = df['abstract'].apply(preprocess_text)

        if 'paper_id' not in df.columns:
            df['paper_id'] = [f"paper_{i}" for i in range(len(df))]

        if clean_stopword:
            self.data = {
                str(row['paper_id']): {
                    "title": str(row['title_clean']),
                    "abstract": str(row.get('abstract_clean', "")),
                }
                for _, row in df.iterrows()
            }
        else:
            self.data = {
                str(row['paper_id']): {
                    "title": str(row['title']),
                    "abstract": str(row.get('abstract', "")),
                }
                for _, row in df.iterrows()
            }

    def __len__(self):
        return len(self.data)

    def batches(self):
        batch = []
        batch_ids = []
        for i, (k, d) in enumerate(self.data.items(), start=1):
            batch_ids.append(k)
            text = d["title"] + " " + (d.get("abstract") or "")
            batch.append(text)

            if i % self.batch_size == 0:
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True,
                    return_tensors="pt", max_length=self.max_length
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                yield inputs, batch_ids
                batch, batch_ids = [], []

        if batch:
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            yield inputs, batch_ids

class Model:
    def __init__(self):
        self.model = AutoModel.from_pretrained('allenai/specter')
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def __call__(self, inputs):
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state[:, 0, :]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='path to CSV file with columns paper_id,title,abstract')
    parser.add_argument('--output', required=True, help='output path (.jsonl)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--clean_stopword', type=bool)
    args = parser.parse_args()

    dataset = Dataset(data_path=args.data_path, batch_size=args.batch_size, clean_stopword=args.clean_stopword)
    model = Model()
    results = {}

    for batch_inputs, batch_ids in tqdm(dataset.batches(), total=len(dataset)//args.batch_size + 1):
        embeddings = model(batch_inputs)
        for paper_id, emb in zip(batch_ids, embeddings):
            results[paper_id] = {
                "paper_id": paper_id,
                "embedding": emb.detach().cpu().numpy().tolist()
            }

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fout:
        for res in results.values():
            fout.write(json.dumps(res) + '\n')

if __name__ == '__main__':
    main()
