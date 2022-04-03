# Incremental Learning
Incremental learning is a method of training a model incrementally by adding new data points to the model. While working with continuous new data, the model is trained on the new data point and the model is updated with the new data point so that it retains the performance on old data.

## How to use
### Setting up the environment
At first create a virtual environment and set up the environment. 
```
conda create --name myenv python=3.8
conda activate myenv
```
Now give run permission to `setup_env.sh` and run from the terminal. This will install all required packages and set up the environment completely.
```
chmod +x setup_env.sh
./setup_env.sh
```