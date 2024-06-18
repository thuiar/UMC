#!/usr/bin/bash

for seed in 0 1 2 3 4
do
    for multimodal_method in 'umc'
    do
        for method in 'umc'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in  'MIntRec' # 'MELD-DA' 'IEMOCAP-DA'
                do
                    python run.py \
                    --dataset $dataset \
                    --data_path 'Datasets' \
                    --logger_name $method \
                    --multimodal_method $multimodal_method \
                    --method $method\
                    --train \
                    --tune \
                    --save_results \
                    --seed $seed \
                    --gpu_id '1' \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_${dataset} \
                    --results_file_name "results_umc.csv" \
                    --output_path "outputs/${dataset}"
                done
            done
        done
    done
done
