GPU=0
DATASET=mm_imdb_cmml
TYPE=both
RATIO=0.7
EXP_NOTE="${DATASET}_${TYPE}_${RATIO}"
TOKENIZERS_PARALLELISM=True python src/main.py experiment=rebq_${DATASET} data=${DATASET} EXP_NOTE=${EXP_NOTE} train.GPU=${GPU} data.missing_params.RATIO=${RATIO} data.missing_params.TYPE=${TYPE} train.EVAL_FREQ=1000 data.NUM_WORKERS=8 test.BATCH_SIZE=4
TOKENIZERS_PARALLELISM=True python src/main.py experiment=rebq_${DATASET} data=${DATASET} EXP_NOTE=${EXP_NOTE} train.GPU=${GPU} data.missing_params.RATIO=${RATIO} data.missing_params.TYPE=${TYPE} train.EVAL_FREQ=1000 data.NUM_WORKERS=8 test.BATCH_SIZE=4 test.TEST_ONLY=True test.CHECKPOINT_DIR=checkpoints/${EXP_NOTE}/checkpoints/
