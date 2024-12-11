
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel  # Replace with the correct import for BGEM3FlagModel

# Global variable for model
model = None

def initialize_bgem3():
    """Initialize the BGEM3 model and tokenizer."""
    global model
    if model is None:
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        model.model.to("cuda:0")

def get_elastic_tokens(es_client, text, tokenizer="standard"):
    """
    Tokenize the text using Elasticsearch's _analyze API.

    :param es_client: Elasticsearch client instance
    :param text: The input text to tokenize
    :param tokenizer: The tokenizer to use (default is "standard")
    :return: List of tokens with their start and end offsets
    """
    response = es_client.indices.analyze(
        body={"tokenizer": tokenizer, "text": text}
    )
    tokens = [
        {"token": t["token"], "start": t["start_offset"], "end": t["end_offset"]}
        for t in response.get("tokens", [])
    ]
    return tokens

def get_bge_token_scores(text):
    """
    Generate token scores using BGEM3 and return token offsets and scores.

    :param text: The input text
    :return: List of dictionaries containing tokens, offsets, and scores
    """
    global model
    initialize_bgem3()
    
    tokenizer = model.tokenizer
    outputs = model.encode(text, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    token_scores = model.convert_id_to_token(outputs["lexical_weights"])
    
    # Get offsets for BGE tokens
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"][1:-1]  # Exclude special tokens ([CLS], [SEP])
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][1:-1])
    
    # Combine tokens, offsets, and scores
    #bge_tokens = [
    #    {"token": tokens[i].lstrip('▁'), "start": offsets[i][0], "end": offsets[i][1], "score": token_scores.get(tokens[i].lstrip('▁'), 0)}
    #    for i in range(len(tokens))
    #]
    bge_tokens = []

    for i in range(len(tokens)):
        token = tokens[i].lstrip('▁')
        start = offsets[i][0]
        end = offsets[i][1]
        if '▁' in tokens[i]:
            start = start + 1
        score = token_scores.get(token, 0)
        bge_tokens.append({"token": token, "start": start, "end": end, "score": score})

    #print(bge_tokens)
    return bge_tokens


def align_scores(es_tokens, bge_tokens, text):
    """
    Align BGE token scores with Elasticsearch tokens based on offsets.

    :param es_tokens: List of Elasticsearch tokens with offsets
    :param bge_tokens: List of BGE tokens with offsets and scores
    :param text: The original text
    :return: List of Elasticsearch tokens with aligned scores
    """
    aligned_tokens = []

    for es_token in es_tokens:
        es_start, es_end = es_token["start"], es_token["end"]
        es_text = text[es_start:es_end]

        # Aggregate scores from BGE tokens overlapping with the Elasticsearch token
        #overlapping_scores = [
        #    bge_token["score"]
        #    for bge_token in bge_tokens
        #    if not (bge_token["end"] <= es_start or bge_token["start"] >= es_end)  # Overlapping condition
        #]
        overlapping_scores = []
        for bge_token in bge_tokens: 
            if bge_token["start"] >= es_start and bge_token["end"] <= es_end:
                overlapping_scores.append(bge_token["score"])

        #print(overlapping_scores)

        token_score = sum(overlapping_scores)  # Aggregate overlapping scores

        aligned_tokens.append({
            "token": es_text,
            "score": token_score,
            "start": es_start,
            "end": es_end
        })
    
    return aligned_tokens

def tokenize_with_scores(es_client, text, tokenizer="standard"):
    """
    Tokenize text using Elasticsearch and align scores from BGEM3.

    :param es_client: Elasticsearch client instance
    :param text: The input text
    :param tokenizer: The tokenizer to use (default is "standard")
    :return: List of Elasticsearch tokens with aligned scores
    """
    # Get Elasticsearch tokens
    es_tokens = get_elastic_tokens(es_client, text, tokenizer)

    # Get BGE tokens with scores
    bge_tokens = get_bge_token_scores(text)

    # Align scores with Elasticsearch tokens
    final_tokens = align_scores(es_tokens, bge_tokens, text)

    return final_tokens

# Example usage
if __name__ == "__main__":
    es_client = Elasticsearch("http://localhost:9200")  # Replace with your Elasticsearch URL
    #text = "This is a test sentence for matching tokens with scores."
    text = "This version avoids Pythonic idioms like comprehensions and provides a step-by-step assignment for clarity."
    

    # Tokenize with scores
    tokens_with_scores = tokenize_with_scores(es_client, text)

    # Print results
    for token in tokens_with_scores:
        #print(f"Token: {token['token']}, Score: {token['score']}, Start: {token['start']}, End: {token['end']}")
        print(f"Token: {token['token']}")
