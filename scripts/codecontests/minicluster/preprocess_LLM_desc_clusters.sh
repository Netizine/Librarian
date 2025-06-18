#!/bin/bash

# Process LLM description clusters 
for i in 0 1 2 3 4 5 6 7 8 9
do
    echo "Preprocessing cluster $i"
 
    uv run python program_refactoring/domains/codecontests_cpp/generate_collection.py --data_file mini_clusters/LLM_description_clusters/new/$i.jsonl --output_dir python_data/mini_clusters/LLM_description_clusters_${i}_ask/my_vectordb
    
    # Archived, ignore
    # uv run python program_refactoring/domains/codecontests_cpp/generate_collection.py --data_file mini_clusters/LLM_description_clusters/new/$i.jsonl --output_dir python_data/mini_clusters/LLM_description_clusters_${i}_problem/my_vectordb
    # uv run python program_refactoring/domains/codecontests_cpp/generate_collection.py --data_file mini_clusters/LLM_description_clusters/new/$i.jsonl --output_dir python_data/mini_clusters/LLM_description_clusters_${i}_solution/my_vectordb
    

done
