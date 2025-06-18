import json
import os

import chromadb
import networkx as nx
import numpy as np
from chromadb.utils import embedding_functions
from sklearn.cluster import AgglomerativeClustering


def get_and_save_embeddings(
    names,
    queries,
    programs,
    ids,
    alltests,
    name,
    persist_directory,
    metadata=None,
    use_query=True,
):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
        # api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
    )
    if use_query:
        # index on the query
        docs = queries
    else:
        # index on the program
        docs = programs

    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
    )

    collection = chroma_client.get_or_create_collection(
        name=name, embedding_function=openai_ef
    )

    # load in /home/zk66/refactor/codecontests/LibLearn/data/codecontests_names_and_short_descriptions.jsonl
    with open(
        # "/home/zk66/refactor/codecontests/LibLearn/data/codecontests_names_and_long_descriptions.jsonl",
        "data/codecontests_names_and_long_descriptions.jsonl",
        "r",
    ) as f:
        lines = f.readlines()

    # lines are each a dictionary of "name" and "short_description"
    # create a dictionary of name to short_description
    name_to_short_description = {}
    for line in lines:
        line = line.strip()
        if line:
            entry = json.loads(line)
            name = entry["name"]
            short_description = entry["short_description"]
            name_to_short_description[name] = short_description

    # # create metadata
    # metadatas = []
    # if metadata is not None:
    #     assert len(metadata) == len(queries)
    #     iterator = zip(queries, programs, alltests, metadata)
    # else:
    #     iterator = zip(queries, programs, alltests)

    # for item in iterator:
    #     if metadata is not None:
    #         query, program, tests, metadata = item
    #         metadatas.append({"query": query, "program": program, "tests": tests, "metadata": metadata})
    #     else:
    #         query, program, tests = item
    #         metadatas.append({"query": query, "program": program, "tests": tests})

    # Create metadata
    metadatas = []
    if metadata is not None:
        assert len(metadata) == len(queries)
        iterator = zip(names, queries, programs, alltests, metadata)
    else:
        iterator = zip(names, queries, programs, alltests)

    for item in iterator:
        if metadata is not None:
            name_, query, program, tests, meta = item
            short_description = name_to_short_description.get(name_, "")
            metadatas.append(
                {
                    "query": query,
                    "program": program,
                    "tests": tests,
                    "metadata": meta,
                    "short_description": short_description,
                }
            )
        else:
            name_, query, program, tests = item
            short_description = name_to_short_description.get(name_, "")
            metadatas.append(
                {
                    "query": query,
                    "program": program,
                    "tests": tests,
                    "short_description": short_description,
                }
            )

    # Build docs_short based on names (NOT ids)
    docs_short = []
    for name_ in names:
        short_description = name_to_short_description.get(name_, "")
        if short_description == "":
            import pdb

            pdb.set_trace()
        docs_short.append(short_description)

    print("docs_short_len", len(docs_short))

    collection.add(documents=docs_short, ids=ids, metadatas=metadatas)
    # collection.add(documents=docs, ids=ids, metadatas=metadatas)

    # Print the collections
    print(f"Collection Name: {collection.name}")
    print(f"Number of entries: {collection.count()}")

    # Retrieve and print the entries in a readable form
    entries = collection.get()
    # print(f"Raw entries data: {entries}")

    # if isinstance(entries, dict):
    #     # Assuming entries is a dictionary with keys 'ids', 'documents', 'metadatas', 'embeddings'
    #     # ids = entries.get('ids', [])
    #     print(entries.keys())
    #     documents = entries.get('documents', [])
    #     metadatas = entries.get('metadatas', [])
    #     embeddings = entries.get('embeddings', [])

    #     if ids:
    #         # first_id = ids[0]
    #         first_document = documents
    #         first_metadata = metadatas
    #         first_embedding = embeddings

    #         print("First Entry:")
    #         # print(f"  ID: {first_id}")
    #         print(f"  Document: {first_document}")
    #         print(f"  Metadata: {first_metadata}")
    #         print(f"  Embedding: {first_embedding}")
    #     else:
    #         print("No entries found.")
    # else:
    #     print("Entries data is not in the expected dictionary format.")
    #     print(f"Entries data: {entries}")


def load_from_dir(persist_directory, name):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        # api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
    )
    collection = chroma_client.get_or_create_collection(
        name=name, embedding_function=openai_ef
    )
    return collection


def cluster_embeddings(embeddings, ids):
    X = np.array(embeddings)
    print("shape of X", X.shape)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1).fit(X)
    return clustering, ids


def create_graph(clustering, ids):
    graph = nx.DiGraph()

    # ids = all_data['ids']
    sample_size = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        for child_idx in merge:
            if child_idx < sample_size:
                # from docs: at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
                node_name = f"{child_idx}:{ids[child_idx]}"
                graph.add_node(node_name)
                graph.add_edge(i + sample_size, node_name)
            else:
                graph.add_edge(i + sample_size, child_idx)
    return graph


def visualize_graph(graph):
    lines = nx.generate_network_text(graph, with_labels=True)
    for line in lines:
        print(line)


def clean_header(header, code):
    len_head = len(header)
    code = code[len_head:].strip()
    return code
