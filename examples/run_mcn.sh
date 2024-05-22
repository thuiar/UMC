#!/usr/bin/bash

for seed in 0 1 2 3 4
do
    for multimodal_method in 'mcn'
    do
        for method in 'mcn'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                python run.py \
                --dataset 'MIntRec' \
                --logger_name $method \
                --multimodal_method $multimodal_method \
                --method $method\
                --train \
                --tune \
                --save_results \
                --save_model \
                --seed $seed \
                --gpu_id '3' \
                --video_feats_path 'video_feats.pkl' \
                --audio_feats_path 'audio_feats.pkl' \
                --text_backbone $text_backbone \
                --config_file_name $method \
                --results_file_name "results_$method.csv" \
                --output_path 'outputs' \
                --data_path 'Datasets/' 
            done
        done
    done
done