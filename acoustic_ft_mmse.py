#!/home/nevikw/miniconda3/envs/ml-project/bin/python

from argparse import ArgumentParser
from random import randint
import warnings

from datasets import load_dataset, Audio, Value
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate


warnings.filterwarnings("ignore")

ap = ArgumentParser()
ap.add_argument("-m", "--base-model", type=str, default="openai/whisper-medium")
ap.add_argument("-d", "--sample-duration", type=int, default=30)
ap.add_argument("-b", "--batch-size", type=int, default=4)
ap.add_argument("-g", "--grad-accu-step", type=int, default=8)

args = ap.parse_args()

feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)

preprocess = lambda examples: feature_extractor(
    [i["array"][(n := randint(0, len(i["array"]) - (m := min(len(i["array"]), feature_extractor.sampling_rate*args.sample_duration)))) : n + m] for i in examples["audio"]],
    sampling_rate=feature_extractor.sampling_rate,
    do_normalize=True,
)

dataset = (
    load_dataset("nevikw39/ADReSSo")
    .cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    .cast_column("mmse", Value("float"))
)
dataset["train"], dataset["valid"] = dataset["train"].train_test_split(0.25).values()

encoded_dataset = dataset.map(preprocess, remove_columns=["audio", "label"], batched=True, load_from_cache_file=False).rename_column("mmse", "label")

model = AutoModelForAudioClassification.from_pretrained(args.base_model, num_labels=1)

mse = evaluate.load("mse")

training_args = TrainingArguments(
    output_dir="models/" + args.base_model[args.base_model.index('/') + 1 :] + "_ADReSSo-MMSE",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=1e-3,
    weight_decay=5e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size*2,
    gradient_accumulation_steps=args.grad_accu_step,
    # gradient_checkpointing=True,
    num_train_epochs=100,
    warmup_ratio=.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,
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
    compute_metrics=lambda eval_pred: mse.compute(
        predictions=eval_pred.predictions,
        references=eval_pred.label_ids,
        squared=False
    ),
    callbacks=[EarlyStoppingCallback(10)],
)

trainer.train()
print(trainer.evaluate(encoded_dataset["test"]))
trainer.save_model("models/" + args.base_model[args.base_model.index('/') + 1 :] + "_ADReSSo-MMSE")
trainer.push_to_hub()
