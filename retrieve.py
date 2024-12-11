from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from elasticsearch import Elasticsearch
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection
from utils import tokenize_with_scores
from beir.retrieval.evaluation import EvaluateRetrieval
import ipdb
import argparse
import pathlib
import os
import numpy as np
from scipy.sparse import csr_array, vstack
import json

# there are some built-in analyzers for several languages, now we use 'en' for English.
analyzer = build_default_analyzer(language="en")

class ElasticTokenizer():
    def __init__(self):
        self.es_client = Elasticsearch("http://localhost:9200")

    def tokenize(self, text):
        tokenizer = "standard"
        response = self.es_client.indices.analyze(
            body={"tokenizer": tokenizer, "text": text}
        )
        tokens = [
            {"token": t["token"], "start": t["start_offset"], "end": t["end_offset"]}
            for t in response.get("tokens", [])
        ]
        return [token['token'] for token in tokens]


def sparse_equal(A, B):
    return (A != B).nnz == 0

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure settings for your application.")

    parser.add_argument(
        "--use_bm25",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--insert",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--keep_stopword",
        type=bool,
        default=False,
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    analyzer = build_default_analyzer(language="en")
    bm25_ef = BM25EmbeddingFunction(analyzer)
    dataset = 'fiqa'

    analyzer2 = build_default_analyzer(language="en")
    bm25_ef2 = BM25EmbeddingFunction(analyzer2)

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    bm25_ef.load("fiqa.json")
    if args.keep_stopword:
        filters = [bm25_ef.analyzer.filters[0], bm25_ef.analyzer.filters[1], bm25_ef.analyzer.filters[-1]]
        bm25_ef.analyzer.filters = filters
    
    db = MilvusClient(uri='temp_test1.db')
    insert = args.insert
    learned_idf = not args.use_bm25

    if insert is True:
        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
	    )
        schema.add_field(
            field_name="cid",
            datatype=DataType.VARCHAR,
            max_length=100,
	    )

        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=1024,
	    )

        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        collection_name = "test"
        db.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True
        )

        for cid in corpus:
            content = corpus[cid]['text']

            doc_emb = bm25_ef.encode_documents([content])
            db.insert(collection_name, {"sparse_vector": doc_emb, "cid": cid, "content": content})

    stemmer = analyzer.filters[-1]

    if learned_idf:
        final_results = {}
        es_client = Elasticsearch("http://localhost:9200")
        for qid in queries:
            tokens = tokenize_with_scores(es_client, queries[qid])
            values = []
            rows = []
            cols = []
            for token in tokens:
                tok = stemmer.apply([token['token']])[0]
                if tok not in bm25_ef.idf:
                    continue
                values.append(token['score'])
                rows.append(0)
                cols .append(bm25_ef.idf[tok][1])
            query_emb = csr_array((values, (rows, cols)), shape=(1, len(bm25_ef.idf))).astype(np.float32)

            #query_emb2_indices = query_emb2.nonzero()[1]
			
            ## Filter query_emb to retain only these indices
            #rows, cols = query_emb.nonzero()  #
            #filtered_rows = []
            #filtered_cols = []
            #filtered_values = []
            #rows, cols = query_emb.nonzero()  #
            
            #for row, col, val in zip(rows, cols, query_emb.data):
            #    if col in query_emb2_indices:  # Check if the column index exists in query_emb2
            #        filtered_rows.append(row)
            #        filtered_cols.append(col)
            #        filtered_values.append(val)
            
            # Create a new sparse array with the filtered data
            #filtered_query_emb = csr_array(
            #    (filtered_values, (filtered_rows, filtered_cols)),
            #    shape=query_emb.shape
            #)
            results = db.search(collection_name='test', data=[query_emb], topk=5, output_fields=['cid'])

            final_results[qid] = {}
            for result in results[0]:
                final_results[qid][result['entity']['cid']] = result['distance']
    else:
        final_results = {}
        for qid in queries:
            final_results[qid] = {}
            query_emb = bm25_ef.encode_queries([queries[qid]])
            results = db.search(collection_name='test', data=[query_emb], topk=5, output_fields=['cid'])
            for result in results[0]:
                final_results[qid][result['entity']['cid']] = result['distance']
    with open('final_results.json', 'w') as fw:
        fw.write(json.dumps(final_results, indent=4))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, final_results, [1,10,100])
    print(ndcg)
