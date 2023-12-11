import argparse
from copy import deepcopy
from random import randint
import warnings

from datasets import load_dataset, concatenate_datasets, Audio, Dataset
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
ap.add_argument("-m", "--base-model", type=str, default="distil-whisper/distil-large-v2")
ap.add_argument("-d", "--sample-duration", type=int, default=30)
ap.add_argument("-b", "--batch-size", type=int, default=2)
ap.add_argument("-g", "--grad-accu-step", type=int, default=32)
ap.add_argument("-n", "--sample-frequence", type=int, default=5) # threshold of an audio
ap.add_argument("-p", "--predict-step", type=int, default=6)     # it will predict 2 * predict_step + 1 rounds
ap.add_argument('-f')

args = ap.parse_args()

PATH = f"models/{args.base_model[args.base_model.index('/') + 1 :]}_ADReSSo"

feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)

train_preprocess = lambda examples: feature_extractor(
    [i["array"][(n := randint(0, len(i["array"]) - (m := min(len(i["array"]), feature_extractor.sampling_rate*args.sample_duration)))) : n + m] for i in examples["audio"]],
    sampling_rate=feature_extractor.sampling_rate,
    do_normalize=True,
    # return_attention_mask=True,
    # max_length=16_000*args.sample_duration,
    # truncation=True,
)

dataset = (
    load_dataset("nevikw39/ADReSSo")
    .with_format("np")
    .cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
)
dataset["train"], dataset["valid"] = dataset["train"].shuffle().train_test_split(0.25).values()
dataset["test"] = dataset["test"].shuffle()

trains = []
for i in dataset["train"]:
    n = min((len(i["audio"]["array"]) + feature_extractor.sampling_rate * args.sample_duration - 1) // (feature_extractor.sampling_rate * args.sample_duration), args.sample_frequence)
    for j in range(n):
        trains.append(deepcopy(i))
dataset["train"] = Dataset.from_list(trains, dataset["train"].features, dataset["train"].info)
encoded_dataset_train = dataset["train"].map(train_preprocess, remove_columns="audio", batched=True)
encoded_dataset = dataset.map(train_preprocess, remove_columns="audio", batched=True)

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
specificity = evaluate.load("nevikw39/specificity")

compute_metrics = lambda eval_pred: (
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
)

training_args = TrainingArguments(
    output_dir="models/" + args.base_model[args.base_model.index("/") + 1 :] + "_ADReSSo_v2",
    # fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=3e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accu_step,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
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
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset["valid"],
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
    callbacks=[EarlyStoppingCallback(10)],
)

trainer.train()
print(trainer.evaluate(encoded_dataset["test"]))
trainer.push_to_hub()

# prediction
pred = np.argmax(trainer.predict(encoded_dataset["test"]).predictions, axis=1)
for i in range(2 * args.predict_step):
  encoded_dataset["test"] = dataset["test"].map(train_preprocess, remove_columns="audio", batched=True, load_from_cache_file=False)
  pred += np.argmax(trainer.predict(encoded_dataset["test"]).predictions, axis=1)
pred[pred <= args.predict_step] = 0
pred[pred > args.predict_step] = 1

eval_pred = trainer.predict(encoded_dataset["test"])
print(f'{accuracy.compute(predictions=pred,references=eval_pred.label_ids)}')
print(f'{f1.compute(predictions=pred,references=eval_pred.label_ids)}')
print(f'{specificity.compute(predictions=pred,references=eval_pred.label_ids)}')