#!/home/nevikw/miniconda3/envs/ml-project/bin/python

import sys
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

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "facebook/wav2vec2-base"

dataset = (
    load_dataset("nevikw39/ADReSSo")
    .shuffle()
    .cast_column("audio", Audio(sampling_rate=16_000))
)
dataset["train"], dataset["valid"] = dataset["train"].train_test_split(0.1).values()

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

encoded_dataset = dataset.map(
    lambda examples: feature_extractor(
        [x["array"] for x in examples["audio"]],
        sampling_rate=feature_extractor.sampling_rate,
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
    MODEL_NAME, num_labels=num_labels, label2id=label2id, id2label=id2label
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

training_args = TrainingArguments(
    output_dir="models/" + MODEL_NAME[MODEL_NAME.index("/") + 1 :] + "_ADReSSo",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=32,
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
    )
    | f1.compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids,
    ),
    callbacks=[EarlyStoppingCallback(10)],
)

trainer.train()
trainer.evaluate(encoded_dataset["test"])
trainer.save_model("models/" + MODEL_NAME[MODEL_NAME.index("/") + 1 :] + "_ADReSSo")
trainer.push_to_hub()
