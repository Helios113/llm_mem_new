import os
from data_formatting import get_tokenizer_and_data_collator_and_prompt_formatting
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,TrainerCallback
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from flwr_datasets import FederatedDataset
from transformers.trainer_utils import EvalPrediction
from flwr_datasets.partitioner import IidPartitioner
import wandb
import numpy 
import torch
from datasets import load_dataset
import copy
from evaluate import load
from rouge_score import rouge_scorer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def gen_compute_metrics(tokenizer):
    
    def compute_metrics(eval_preds):
        # Get predictions and labels from EvalPrediction object
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

        # Reshape if needed - predictions are [batch_size, seq_length, vocab_size]
        if len(predictions.shape) == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            labels = labels.reshape(-1)

        # Convert predictions and labels to PyTorch tensors
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)

        # Calculate cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions, labels)

        # Calculate perplexity as exp(loss)
        perplexity = torch.exp(loss)
        pred_ids = predictions.argmax(dim=-1)
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        labels[labels == -100] = tokenizer.pad_token_id
        

        

        # Decode predictions and labels into text
        decoded_preds = tokenizer.decode(pred_ids , skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
        # print("decoded preds",decoded_preds)
        # print("decoded labels",decoded_labels)
        # print("pred_ids",type(decoded_preds))
        # print("pred_ids",type(decoded_labels))
        # print("pred_ids",len(decoded_preds))
        # print("pred_ids",len(decoded_labels))
        # # Load ROUGE metric
        # exit()
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True, tokenizer=tokenizer)
        
        scores = scorer.score(decoded_preds, decoded_labels)["rouge1"]
        print(scores)
        # print(scores)
        print(scores.fmeasure)
        
        return {"perplexity": perplexity.item(), "rouge1": scores.fmeasure}
    
    return compute_metrics


# wandb.init(project="federated-flwr", name=f"master")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=None,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer, collator = get_tokenizer_and_data_collator_and_prompt_formatting("EleutherAI/pythia-14m", "EleutherAI/pythia-14m")
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")

model = get_peft_model(base_model, lora_config)

training_args = SFTConfig(
    output_dir="/tmp",
    dataset_text_field="text",
    save_steps=10,
    logging_steps=2,
    max_steps=4,
    report_to="none",
    per_device_train_batch_size=4,  # Reduce batch size
)

partitioner = IidPartitioner(num_partitions=8)

fds = FederatedDataset(
    dataset="json",
    partitioners={"train": partitioner},
    data_files="/nfs-share/pa511/llm_memorisation/datasets_our/medical_dataset/deduplicated_medical_meadow_flashcards_train.json",
)
train_data = fds.load_partition(0, "train")
val_set = load_dataset(
        "json",
        data_files="/nfs-share/pa511/llm_memorisation/datasets_our/medical_dataset/deduplicated_medical_meadow_flashcards_non_member.json",
        split="train",
    )
class GlobalStepCallback(TrainerCallback):
    def __init__(self, current_round, steps_per_round, is_training=True):
        self.current_round = current_round
        self.steps_per_round = steps_per_round
        self.is_training = is_training

    def on_log(self, args, state, control, logs=None, **kwargs):
        # print(state.log_history)
        cur_step = (self.current_round - 1) * self.steps_per_round + state.log_history[
            -1
        ]["step"]
        print("CURESTEP before:",cur_step)
        if self.is_training and state.log_history[-1]["step"] == 0:
            cur_step+=1
        print("CURESTEP after:",cur_step)
        commit_dict = {}
        for i in state.log_history[-1]:
            if i.startswith("train_"):
                key = copy.copy(i).replace("train_", "train/")
                commit_dict[key] = state.log_history[-1][i]
            elif not i.startswith("eval_"):
                key = "train/" + i
                commit_dict[key] = state.log_history[-1][i]
            print(commit_dict)
        # wandb.log(data=commit_dict, step=cur_step)

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        cur_step = (self.current_round - 1) * self.steps_per_round + state.log_history[
            -1
        ]["step"]
        print(cur_step)
        commit_dict = {}
        for i in state.log_history[-1]:
            if i.startswith("eval_"):
                key = copy.copy(i).replace("eval_", "eval/")
                commit_dict[key] = state.log_history[-1][i]
            else:
                key = "eval/" + i
                commit_dict[key] = state.log_history[-1][i]
            print(commit_dict)
        # wandb.log(data=commit_dict, step=cur_step)
for i in range(1,4):
    print(i)
    global_step_callback = GlobalStepCallback(
            current_round=i, steps_per_round=4
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=val_set,
        data_collator=collator,
        processing_class = tokenizer,
        compute_metrics=gen_compute_metrics(tokenizer),
        callbacks=[global_step_callback],
    )   
            
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    trainer.train()
    # # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # torch.cuda.empty_cache()

# wandb.finish()
