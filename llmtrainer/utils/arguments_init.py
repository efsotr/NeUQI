import sys

from transformers import HfArgumentParser, LlamaTokenizer

from .arguments import DataTrainingArguments, MaxTrainingArguments, ModelArguments


MODEL_ARGS : ModelArguments = None
DATA_ARGS : DataTrainingArguments = None
TRAINING_ARGS : MaxTrainingArguments = None
TOKENIZER : LlamaTokenizer = None

def init_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MaxTrainingArguments))
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        if config_path.endswith(".json"):
            args = parser.parse_json_file(config_path)
        elif config_path.endswith(".yaml"):
            args = parser.parse_yaml_file(config_path)
        else:
            raise NotImplementedError(config_path)
    else:
        args = parser.parse_args_into_dataclasses()

    global MODEL_ARGS, DATA_ARGS, TRAINING_ARGS
    MODEL_ARGS = args[0]
    DATA_ARGS = args[1]
    TRAINING_ARGS = args[2]

def get_model_args():
    global MODEL_ARGS
    return MODEL_ARGS

def get_data_args():
    global DATA_ARGS
    return DATA_ARGS

def get_training_args():
    global TRAINING_ARGS
    return TRAINING_ARGS

def set_tokenizer(tokenizer):
    global TOKENIZER
    TOKENIZER = tokenizer

def get_tokenizer():
    global TOKENIZER
    return TOKENIZER
