# coding=utf-8

import importlib
import logging
import sys

import torch
import transformers
from process_data import get_data

from utils import (
    EndEvalCallback,
    get_data_args,
    get_model_args,
    get_training_args,
    init_args,
    load_model,
    load_tokenizer,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    init_args()

    training_args = get_training_args()

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"check memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3} GB")
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {get_data_args()}")
    logger.info(f"Model parameters {get_model_args()}")

    # load tokenizer
    load_tokenizer()

    # Get the datasets
    train_dataset, eval_datasets, data_collator = get_data()
    logger.info('Finish loading dataset')

    # if training_args.do_eval:
    #     assert eval_datasets is not None
    # else:
    #     eval_datasets = None

    # Load model
    model = load_model()

    module = importlib.import_module("utils.trainer_" + training_args.training_type)
    trainer_class = module.Trainer
    metric_class = module.Metric

    # Initialize our Trainer
    trainer = trainer_class(
        model=model,
        processing_class=None,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        compute_metrics=metric_class(),
        callbacks=[EndEvalCallback],
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()

