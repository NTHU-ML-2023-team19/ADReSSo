#!/home/nevikw/miniconda3/envs/ml-project/bin/python

from argparse import ArgumentParser
from copy import deepcopy
from random import randint
import warnings

from datasets import load_dataset, Audio, Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np
import librosa


warnings.filterwarnings("ignore")

ap = ArgumentParser()
ap.add_argument("-m", "--base-model", type=str, default="whisper-large-v3")
ap.add_argument("-d", "--sample-duration", type=int, default=30)
ap.add_argument("-b", "--batch-size", type=int, default=8)
ap.add_argument("-g", "--grad-accu-step", type=int, default=4)

args = ap.parse_args()

feature_extractor = AutoFeatureExtractor.from_pretrained(f"./models/{args.base_model}_ADReSSo")

preprocess = lambda examples: feature_extractor(
    [i["array"][(n := randint(0, len(i["array"]) - (m := min(len(i["array"]), feature_extractor.sampling_rate*args.sample_duration)))) : n + m] for i in examples["audio"]],
    sampling_rate=feature_extractor.sampling_rate,
    do_normalize=True,
)

valid = []
for i in load_dataset("NTHU-ML-2023-team19/ADReSS-M", split="validation").cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate)):
    for j in np.array_split(i["audio"]["array"], (len(i["audio"]["array"]) + feature_extractor.sampling_rate * 30 - 1) // (feature_extractor.sampling_rate * 30)):
        dc = deepcopy(i)
        dc["audio"]["array"] = j
        valid.append(dc)
        dc = deepcopy(i)
        dc["audio"]["array"] = j + .005 * np.random.uniform() * np.amax(j) * np.random.normal(size=j.shape[0])
        valid.append(dc)
valid = Dataset.from_list(valid).map(preprocess, remove_columns="audio", batched=True, load_from_cache_file=False)
test = load_dataset("NTHU-ML-2023-team19/ADReSS-M", split="test").cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate)).map(preprocess, remove_columns="audio", batched=True, load_from_cache_file=False)

labels = test.features["label"].names
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model = AutoModelForAudioClassification.from_pretrained(
    f"./models/{args.base_model}_ADReSSo", num_labels=num_labels, label2id=label2id, id2label=id2label
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
specificity = evaluate.load("nevikw39/specificity")

training_args = TrainingArguments(
    output_dir="models/" + args.base_model + "_ADReSS-M_transfer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=3e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accu_step,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=10,
    warmup_ratio=.2,
    logging_steps=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub_organization="NTHU-ML-2023-team19",
    push_to_hub=True,
    hub_private_repo=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=valid,
    eval_dataset=test,
    # eval_dataset=encoded_dataset["test"].select(np.random.choice(71, 42)),
    tokenizer=feature_extractor,
    compute_metrics=lambda eval_pred: (
        accuracy.compute(
            predictions=(pred := np.argmax(eval_pred.predictions, axis=1)),
            references=eval_pred.label_ids,
        ) | f1.compute(
            predictions=pred,
            references=eval_pred.label_ids,
        ) | specificity.compute(
            predictions=pred,
            references=eval_pred.label_ids,
        )
    ),
    # callbacks=[EarlyStoppingCallback(10)],
)

trainer.train()

print(trainer.evaluate())

trainer.save_model("models/" + args.base_model + "_ADReSS-M")
trainer.push_to_hub()
