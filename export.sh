python3 export.py \
        --weights weights/Animal_x-d-416-768_20230510.pt \
        --img-size 416 768 \
        --batch-size 1 \
        --iou-thres 0.45 \
        --conf-thres 0.25 \
        --simplify

