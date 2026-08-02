from pathlib import Path
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tokenizer = Tokenizer.from_files(
    vocab_filepath=str(PROJECT_ROOT / "tinystories_vocab.pkl"),
    merges_filepath=str(PROJECT_ROOT / "tinystories_merges.pkl"),
    special_tokens=["<|endoftext|>"],
)

print("vocabulary size", len(tokenizer.vocab))
print("number of merges", len(tokenizer.merges))

with (PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt").open(
    "r", encoding="utf-8"
) as f:
    text = f.read()


documents = text.split("<|endoftext|>")
sample_document = documents[:10]

sample_text = "<|endoftext|>".join(sample_document)

token_ids = tokenizer.encode(sample_text)
num_bytes = len(sample_text.encode("utf-8"))
num_tokens = len(token_ids)
compression_ratio = num_bytes / num_tokens

print("Bytes", num_bytes)
print("Tokens", num_tokens)
print("Compression ratio", compression_ratio)

total_bytes = len(text.encode("utf-8"))

start_time = time.perf_counter()
all_token_ids = tokenizer.encode(text)
elapsed_seconds = time.perf_counter() - start_time

bytes_per_second = total_bytes / elapsed_seconds
megabytes_per_second = bytes_per_second / 1_000_000

# estimate how long it would take to token
pile_bytes = 825 * 1_000_000_000
estimated_seconds = pile_bytes / bytes_per_second
estimated_hours = estimated_seconds / 3600
estimated_days = estimated_hours / 24

print("throughput test bytes:", total_bytes)
print("elapsed seconds:", elapsed_seconds, "seconds")
print("throughput", bytes_per_second, "bytes/second")
print("throughput", megabytes_per_second, "megabytes/second")
print("estimated pile time:", estimated_hours, "hours, ", estimated_days, "days")

def encode_and_save(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as file:
        token_ids = np.fromiter(
            tokenizer.encode_iterable(file),
            dtype=np.uint16,
        )
        np.save(output_path, token_ids)
        print("saved: ", output_path)
        print("number of tokens: ", len(token_ids))
        print("datatype: ", token_ids.dtype)

encode_and_save(
    PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
    PROJECT_ROOT / "data" / "tinystories_train_tokens.npy",
)

encode_and_save(
    PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
    PROJECT_ROOT / "data" / "tinystories_valid_tokens.npy",
)
