#!/bin/bash
WORKING_DIR=$(pwd)

if [ "$1" == "search" ] ; then
    rm -r search-EXP-*
    echo "Starting Search"
    python ./cnn/train_search.py --unrolled --batch_size=2 --report_freq=1 ${*:2}

elif [ "$1" == "train" ] ; then
    echo "Starting Training the full model"
    python ./cnn/final_model/train.py --report_freq 1 ${*:2}

elif [ "$1" == "predict" ] ; then
    echo "Starting Predictions"
    python ./cnn/predict.py ${*:2}

elif [ "$1" == "vizualize" ] | [ "$1" == "viz" ] | [ "$1" == "vizualise" ] ; then
    echo "Making cell structures"
    python .\cnn\final_model\visualize.py ${*:2}
    echo "Navigate to $WORKING_DIR/cnn/final_model/cells"

else
    echo "no command $1 found! Use search/train/predict/vizualise"
fi