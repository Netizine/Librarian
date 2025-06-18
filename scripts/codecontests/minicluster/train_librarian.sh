#!/bin/bash

mkdir -p train_metrics

SAMPLE_K=8
MAX_TUPLE_SIZE=3

# Define clusters to process 
for i in 0 1 2 3 4 5 6 7 8 9 
do
    echo "--------------------------------------------------"
    echo "Training ReGAL on cluster $i with sample budget = ${SAMPLE_K} (ask)"
    echo "--------------------------------------------------"

    # Define unique temp directory including sample_budget
    TEMP_DIR="temp_new_${i}_sample_budget${SAMPLE_K}"

    # Define unique output file name including sample_k
    OUTPUT_FILE="train_metrics/librarian_metrics_minicluster${i}_sample_budget_${SAMPLE_K}.txt"

    # Run training and capture output
    TRAINING_OUTPUT=$(uv run python program_refactoring/refactor_db.py \
        --collection_path python_data/mini_clusters/LLM_description_clusters_${i}_ask/my_vectordb/ \
        --model_name o4-mini \
        --reset_codebank \
        --retrieval_method ask \
        --filter_every 100 \
        --refactor_every 5 \
        --task python \
        --dataset code_contests \
        --tree_type big_tree \
        --max_tuple_size ${MAX_TUPLE_SIZE} \
        --do_retry \
        --helpers_second \
        --temp_dir ${TEMP_DIR} \
        --sample_k ${SAMPLE_K})

    # Extract log directory from output
    LOG_DIR=$(echo "$TRAINING_OUTPUT" | grep "LOG_DIR_PATH:" | sed 's/LOG_DIR_PATH://')

    # Check if LOG_DIR was successfully extracted
    if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then # Also check if it's a directory
        echo "Extracted log directory: $LOG_DIR"

        # run eval using the extracted log directory
        echo "Evaluating ReGAL on new cluster $i with sample_k ${SAMPLE_K} (ask)"
        uv run python scripts/codecontests/postprocess_metrics.py \
            --log_dir "$LOG_DIR" > "$OUTPUT_FILE"

        echo "Evaluation results saved to $OUTPUT_FILE"

    elif [ -n "$LOG_DIR" ]; then
            echo "Warning: Extracted LOG_DIR '$LOG_DIR' is not a valid directory. Evaluation skipped."
            echo "Full training output:"
            echo "$TRAINING_OUTPUT"
    else
        echo "Failed to extract log directory from training output. Evaluation skipped for new cluster $i, sample_k ${SAMPLE_K}."
        echo "Full training output:"
        echo "$TRAINING_OUTPUT" 
    fi
    echo 
done 

echo "--------------------------------------------------"
echo "All processing complete for sample_k ablation."
echo "--------------------------------------------------"



