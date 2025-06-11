from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy

from .arguments import MaxTrainingArguments


class EndEvalCallback(TrainerCallback):
    def on_step_end(self, args: MaxTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log & Evaluate & Save
        if state.global_step >= state.max_steps:
            if args.save_strategy == IntervalStrategy.STEPS:
                control.should_save = True
            if args.eval_strategy == IntervalStrategy.STEPS:
                control.should_evaluate = True

        # if args.check_stage == "ck_run":
        #     if state.global_step >= 3:
        #         control.should_training_stop = True
        #         control.should_save = False
        #         if args.eval_strategy != IntervalStrategy.NO:
        #             control.should_evaluate = True

        return control
