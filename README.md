
# README

## Overview

This script is designed to identify relevant articles to link in a dedicated content block. By leveraging various advanced natural language processing (NLP) techniques, the script finds and recommends similar articles based on their titles and meta descriptions.

## Script Logic

### 1. Loading Data

The script begins by loading data from a CSV file. This file contains article titles and meta descriptions:

```python
file_path = '/Users/simonazoulay/Similar-Content/List-articles-title-meta.csv'
data = pd.read_csv(file_path, delimiter=';')
```

### 2. Combining Title and Meta Description

The title and meta description of each article are combined into a single text field to enhance the semantic understanding:

```python
data['combined'] = data['Title 1'] + " " + data['Meta Description 1']
```

### 3. Similarity Calculation Methods

The script employs five different methods to calculate the similarity between articles:

#### Method 1: SBERT

```python
def find_similar_sbert(data):
    ...
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    ...
```

- **Description**: Uses the Sentence-BERT model, which is an adaptation of BERT designed for generating sentence embeddings.
- **Advantages**: Efficient, multilingual, and provides high-quality embeddings for sentence-level similarity.
- **Disadvantages**: Requires more computational resources compared to simpler models.

#### Method 2: DistilBERT

```python
def find_similar_distilbert(data):
    ...
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    ...
```

- **Description**: DistilBERT is a smaller, faster, cheaper, and lighter version of BERT.
- **Advantages**: Faster and less resource-intensive than BERT while retaining good performance.
- **Disadvantages**: Slightly lower accuracy compared to the full BERT model.

#### Method 3: RoBERTa

```python
def find_similar_roberta(data):
    ...
    model = SentenceTransformer('stsb-roberta-large')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    ...
```

- **Description**: RoBERTa is an optimized version of BERT with improved training procedures.
- **Advantages**: Offers improved performance over BERT.
- **Disadvantages**: More computationally intensive than DistilBERT.

#### Method 4: BERT

```python
def find_similar_bert(data):
    ...
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(data['combined'].tolist(), convert_to_tensor=True)
    ...
```

- **Description**: BERT (Bidirectional Encoder Representations from Transformers) is designed to understand the context of a word in search queries.
- **Advantages**: Provides high-quality sentence embeddings and captures context effectively.
- **Disadvantages**: Resource-intensive and slower compared to DistilBERT.

#### Method 5: USE (Universal Sentence Encoder)

```python
def find_similar_use(data):
    ...
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = model(data['combined'].tolist())
    ...
```

- **Description**: USE provides embeddings for sentences that can be used for text classification, semantic similarity, clustering, and other NLP tasks.
- **Advantages**: Easy to use, fast, and performs well on various tasks.
- **Disadvantages**: May not perform as well as fine-tuned transformer models on specific tasks.

### 4. Generating Results

After calculating cosine similarities using each method, the script generates a list of similar articles for each original article:

```python
def generate_results(data, cosine_similarities, method_name):
    ...
    for idx in range(len(data)):
        similar_indices = get_similar_articles(idx, cosine_similarities)
        similar_urls = data.iloc[similar_indices]['Original Url'].tolist()
        results_list.append({
            'Original Url': data.iloc[idx]['Original Url'],
            'Similar Urls': similar_urls
        })
    ...
```

### 5. Execution

The script executes each method sequentially, calculates similarities, and saves the results:

```python
if __name__ == '__main__':
    ...
    cosine_similarities_sbert = find_similar_sbert(data)
    results_sbert = generate_results(data, cosine_similarities_sbert, 'sbert')
    ...
```

## Conclusion

This script leverages various state-of-the-art NLP models to find similar articles based on their titles and meta descriptions. Each method has its own strengths and weaknesses, providing a balance between accuracy and computational efficiency. By using these methods, you can effectively enhance content interlinking on your platform, improving user engagement and SEO performance.
