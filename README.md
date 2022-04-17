# Incremental Learning
Incremental learning is a method of training a model incrementally by adding new data points to the model. While working with continuous new data, the model is trained on the new data point and the model is updated with the new data point so that it retains the performance on old data.
 
## How to use
### Project structure
```bash
├── data  
├── incremental_asr  
|   ├── configs  
│   ├── modules    
│   └── utils 
├── jsons
└── results
```

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
### Get you data
Get your data in the `data` directory. The data should be in the following format:
```bash
data
├── task_1
│   ├── utterances
│   └── transcripts.tsv
├── task_2
│   ├── utterances
│   └── transcripts.tsv
├── ...
```
The `utterances` directory should contain the audio files and the transcripts.tsv file should contain the transcriptions with `utt_id`, `spk_id`, and `transcript`.

### The initial training
The first step is to train the model on one of the task data. The `train.py` script is used to train the model. Define the data directory in the `incremental_asr/configs/config.yaml` file and run the script.
```bash
cd incremental_asr
python train.py configs/config.yaml
```

### The incremental training
The second step is to train the model on the new data. The `incremental_train.py` script is used to train the model. Define the data directory for your new task in the `incremental_asr/configs/incremental_config.yaml` file and run the script. You can choose one of the following methods to train the model in incremental manner:
- `ft` (fine tuning)
- `rbkd` (using *Response-based knowledge distillation*)
- `ebkd` (using *Explanation-based knowledge distillation*)
- `ts` (teacher-student architecture with both *rbkd* and *ebkd*)

```bash
python incremental_train.py configs/config.yaml --training_type=ts
```
You can use the `--training_type` argument to choose the training method. This can be defined in the configuration file also.