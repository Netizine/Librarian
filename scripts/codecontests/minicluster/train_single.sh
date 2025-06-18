CLUSTER_ID=0

uv run python program_refactoring/refactor_db.py \
            --collection_path python_data/mini_clusters/LLM_description_clusters_${CLUSTER_ID}_ask/my_vectordb/ \
            --model_name o4-mini \
            --reset_codebank \
            --retrieval_method ask \
            --filter_every 100 \
            --refactor_every 5 \
            --task python \
            --dataset code_contests \
            --tree_type big_tree \
            --max_tuple_size 3 \
            --do_retry \
            --helpers_second \
            --temp_dir temp_single_cluster_${CLUSTER_ID} \
            --sample_k 8
