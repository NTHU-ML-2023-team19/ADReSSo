#!/home/nevikw/miniconda3/envs/ml-project/bin/python

from argparse import ArgumentParser
from warnings import filterwarnings
from os import environ

import torch
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
# from peft import (
#     LoraConfig,
#     TaskType,
#     get_peft_model,
# )
import evaluate
import numpy as np


filterwarnings("ignore")

ap = ArgumentParser()
ap.add_argument("-a", "--acoustic-model", type=str, default="openai/whisper-large-v3")
ap.add_argument("-l", "--linguistic-model", type=str, default="bert-large-uncased")
ap.add_argument("-d", "--sample-duration", type=int, default=30)
ap.add_argument("-b", "--batch-size", type=int, default=8)
ap.add_argument("-g", "--grad-accu-step", type=int, default=4)

args = ap.parse_args()

transcriber = pipeline(model=args.acoustic_model, device=int(environ["LOCAL_RANK"]), batch_size=32)
tokenizer = AutoTokenizer.from_pretrained(args.linguistic_model, trust_remote_code=True, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

preprocess = lambda examples: tokenizer(
    [i["text"] for i in transcriber(examples["audio"], chunk_length_s=args.sample_duration)],
    truncation=True,
    return_token_type_ids=False,
)

dataset = load_dataset("nevikw39/ADReSSo")
dataset["train"], dataset["valid"] = dataset["train"].train_test_split(.25).values()

encoded_dataset = dataset.map(preprocess, remove_columns=["audio", "mmse"], batched=True)

del transcriber
torch.cuda.empty_cache()

labels = dataset["train"].features["label"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model = AutoModelForSequenceClassification.from_pretrained(
    args.linguistic_model, num_labels=num_labels, label2id=label2id, id2label=id2label, trust_remote_code=True, ignore_mismatched_sizes=True #, torch_dtype=torch.float16
)

# lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, lora_dropout=.1)
# model = get_peft_model(model, lora_config)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
specificity = evaluate.load("nevikw39/specificity")

training_args = TrainingArguments(
    output_dir="models/" + args.linguistic_model[args.linguistic_model.find('/') + 1 :] + "_ADReSSo",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=3e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size*2,
    gradient_accumulation_steps=args.grad_accu_step,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=100,
    warmup_ratio=.05,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub_organization="NTHU-ML-2023-team19",
    push_to_hub=True,
    hub_private_repo=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda eval_pred: (
        accuracy.compute(
            predictions=(pred := np.argmax(eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions, axis=1)),
            references=eval_pred.label_ids,
        ) | f1.compute(
            predictions=pred,
            references=eval_pred.label_ids,
        ) | specificity.compute(
            predictions=pred,
            references=eval_pred.label_ids,
        )
    ),
    callbacks=[EarlyStoppingCallback(10)],
)

trainer.train()

print(trainer.evaluate(encoded_dataset["test"]))

trainer.save_model("models/" + args.linguistic_model[args.linguistic_model.find('/') + 1 :] + "_ADReSSo")
trainer.push_to_hub()
