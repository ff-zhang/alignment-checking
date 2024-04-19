#!/bin/bash

# Define the path to your test data and script
TEST_DATA_PATH="./glove/kmeans_clusters_500.pkl"
SCRIPT_PATH="lrp_new.py"

# Declare an associative array to map classes to their respective model file paths
declare -A class_model_paths
class_model_paths[0]="models_0_25.pkl"
class_model_paths[1]="models_0_25.pkl"
class_model_paths[6]="models_0_25.pkl"
class_model_paths[12]="models_0_25.pkl"
class_model_paths[16]="models_0_25.pkl"
class_model_paths[176]="models_175_200.pkl"
class_model_paths[492]="models_475_500.pkl"

# Array of target classes
declare -a classes=(0 1 6 12 16 176 492)

# Loop through each class and execute the Python script
for index in "${classes[@]}"
do
   MODELS_PATH="${class_model_paths[$index]}"  # Get the correct model path for each class
   python "$SCRIPT_PATH" --models_path "$MODELS_PATH" --index "$index" --test_data_path "$TEST_DATA_PATH"
done