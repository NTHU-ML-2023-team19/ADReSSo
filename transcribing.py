#!/home/nevikw/miniconda3/envs/ml-project/bin/python

from random import randrange
from argparse import ArgumentParser
from warnings import filterwarnings

from torch.cuda import device_count

from datasets import load_dataset
from transformers import pipeline


filterwarnings("ignore")

ap = ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="openai/whisper-large-v3")
ap.add_argument("-b", "--batch-size", type=int, default=32)

args = ap.parse_args()

transcriber = pipeline(
    model=args.model, device=randrange(device_count()), batch_size=args.batch_size
)

dataset = load_dataset("nevikw39/ADReSSo")

for split in dataset:
    dataset[split] = dataset[split].add_column(
        "transcript_no-chunked",
        [i["text"] for i in transcriber(dataset[split]["audio"], chunk_length_s=0)],
    ).add_column(
        "transcript_chunked",
        [i["text"] for i in transcriber(dataset[split]["audio"], chunk_length_s=30)],
    ).remove_columns("audio")

print(dataset)

dataset.push_to_hub(
    f"ADReSSo_{args.model[args.model.find('/') + 1 :]}_transcript",
    private=True,
)
