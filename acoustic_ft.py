#!/home/nevikw/miniconda3/envs/ml-project/bin/python

import argparse
from random import randint
import warnings

from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np


warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--base-model", type=str, default="facebook/wav2vec2-base")
ap.add_argument("-d", "--sample-duration", type=int, default=30)
ap.add_argument("-b", "--batch-size", type=int, default=8)
ap.add_argument("-g", "--grad-accu-step", type=int, default=4)

args = ap.parse_args()

dataset = load_dataset("nevikw39/ADReSSo").cast_column(
    "audio", Audio(sampling_rate=16_000)
)
dataset["train"], dataset["valid"] = dataset["train"].train_test_split(0.25).values()

feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)

encoded_dataset = dataset.map(
    lambda examples: feature_extractor(
        [x["array"][(y := randint(0, len(x["array"]) - (z := min(len(x["array"]), 16_000*args.sample_duration)))) : y + z] for x in examples["audio"]],
        sampling_rate=feature_extractor.sampling_rate,
        # max_length=16_000*args.sample_duration,
        # truncation=True,
    ),
    remove_columns="audio",
    batched=True,
)

labels = dataset["train"].features["label"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model = AutoModelForAudioClassification.from_pretrained(
    args.base_model, num_labels=num_labels, label2id=label2id, id2label=id2label
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
specificity = evaluate.load("recall") # Specificity is actually recall of the negative class

training_args = TrainingArguments(
    output_dir="models/" + args.base_model[args.base_model.index("/") + 1 :] + "_ADReSSo",
    # fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=3e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accu_step,
    # gradient_checkpointing=True,
    num_train_epochs=100,
    warmup_ratio=0.1,
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
    tokenizer=feature_extractor,
    compute_metrics=lambda eval_pred: accuracy.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
    ) | f1.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
    ) | specificity.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
        pos_label=0,
    ),
    callbacks=[EarlyStoppingCallback(10)],
)

trainer.train()
trainer.evaluate(encoded_dataset["test"])
trainer.save_model("models/" + args.base_model[args.base_model.index("/") + 1 :] + "_ADReSSo")
trainer.push_to_hub()
