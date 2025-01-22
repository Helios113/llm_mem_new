from transformers.trainer_utils import EvalPrediction
import wandb
import torch
import numpy as np
import random
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import copy
import os
import json
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score
# Fix random seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def gen_compute_metrics(tokenizer):
    perplexity_results = []
    rouge1_results = []
    f1_results = []
    precision_results = []
    recall_results = []
    def compute_metrics(eval_preds: EvalPrediction, compute_result):
        nonlocal perplexity_results
        nonlocal rouge1_results
        nonlocal f1_results
        nonlocal precision_results
        nonlocal recall_results
        # Get predictions and labels from EvalPrediction object
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids
        shift_logits = predictions[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        
        # Reshape if needed - predictions are [batch_size, seq_length, vocab_size]
        if len(shift_logits.shape) == 3:
            shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
            labels = labels.reshape(-1)

        # Convert to torch tensors
        # predictions = torch.tensor(predictions)
        # labels = torch.tensor(labels)

        # Calculate cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, labels)

        # Calculate perplexity as exp(loss)
        perplexity = torch.exp(loss)
        
        pred_ids = shift_logits.argmax(dim=-1)
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        labels[labels == -100] = tokenizer.pad_token_id
        
        # Decode predictions and labels into text
        decoded_preds = tokenizer.decode(pred_ids , skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)

        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True, tokenizer=tokenizer)
        
        scores = scorer.score(decoded_preds, decoded_labels)["rouge1"]
        f1 = f1_score(labels.cpu().numpy(), pred_ids.cpu().numpy(), average='weighted', zero_division=0)
        precision = precision_score(labels.cpu().numpy(), pred_ids.cpu().numpy(), average='weighted', zero_division=0)
        recall = recall_score(labels.cpu().numpy(), pred_ids.cpu().numpy(), average='weighted', zero_division=0)
        
        perplexity_results.append(perplexity.item())
        rouge1_results.append(scores.fmeasure)
        f1_results.append(f1)
        precision_results.append(precision)
        recall_results.append(recall)
        
        if compute_result:
            perplexity_mean = np.mean(perplexity_results)
            rouge1_mean = np.mean(rouge1_results)
            f1_mean = np.mean(f1_results)
            precision_mean = np.mean(precision_results)
            recall_mean = np.mean(recall_results)
            perplexity_results = []
            rouge1_results = []
            f1_results = []
            precision_results = []
            recall_results = []
            return {
                "perplexity": perplexity_mean,
                "rouge1": rouge1_mean,
                "f1": f1_mean,
                "precision": precision_mean,
                "recall": recall_mean,
            }
        return {
            "perplexity": perplexity.item(),
            "rouge1": scores.fmeasure,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    
    return compute_metrics
class GlobalStepCallback(TrainerCallback):
    def __init__(self, elapsed_steps, client_id, is_training=True):
        self.elapsed_steps = elapsed_steps
        self.client_id = client_id
        self.is_training = is_training
        self.log_data = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Accumulate logs in JSON structure
        cur_step = self.elapsed_steps + state.log_history[-1]["step"]
        if self.is_training and state.log_history[-1]["step"] == 0:
            cur_step += 1
        commit_dict = {}
        for i in state.log_history[-1]:
            if i.startswith("train_"):
                key = copy.copy(i).replace("train_", "train/")
                commit_dict[key] = state.log_history[-1][i]
            elif not i.startswith("eval_"):
                key = "train/" + i
                commit_dict[key] = state.log_history[-1][i]
        wandb.log(data=commit_dict, step=cur_step)
        self.log_data["train"].append({"cur_step": self.elapsed_steps + state.log_history[-1]["step"], **commit_dict})

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        # Accumulate logs in JSON structure
        cur_step = self.elapsed_steps + state.log_history[-1]["step"]
        if self.is_training and state.log_history[-1]["step"] == 0:
            cur_step += 1
        commit_dict = {}
        for i in state.log_history[-1]:
            if i.startswith("eval_"):
                key = copy.copy(i).replace("eval_", "eval/")
                commit_dict[key] = state.log_history[-1][i]
            else:
                key = "eval/" + i
                commit_dict[key] = state.log_history[-1][i]
        wandb.log(data=commit_dict, step=cur_step)
        self.log_data["eval"].append({"cur_step": self.elapsed_steps + state.log_history[-1]["step"], **commit_dict})

    def save_logs(self, log_dir):
        # Save accumulated logs to a JSON file
        log_file = os.path.join(log_dir, f"logs_{self.client_id}.json")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                existing_data = json.load(f)
            # Merge existing data with new log_data
            for key in self.log_data:
                if key in existing_data:
                    existing_data[key].extend(self.log_data[key])
                else:
                    existing_data[key] = self.log_data[key]
            merged_data = existing_data
        else:
            merged_data = self.log_data

        with open(log_file, "w") as f:
            json.dump(merged_data, f, indent=4)
