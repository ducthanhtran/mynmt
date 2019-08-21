#!/bin/bash
DATA_DIR=~/git/sockeye_document_level_context/testing_data/
TRAIN_SRC=${DATA_DIR}/train.tiny.bpe.de-en.en.gz
TRAIN_TAR=${DATA_DIR}/train.tiny.bpe.de-en.de.gz

OUTPUT=/u/tran/debug/mynmt/

rm -r ${OUTPUT}

source /work/smt2/tran/virtualenvs/mynmt/bin/activate

python3 -m mynmt train \
    --train-source ${TRAIN_SRC} \
    --train-target ${TRAIN_TAR} \
    --output-dir ${OUTPUT}

deactivate
