import os

from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab(special_tokens_bytes: list[bytes]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    cur_id = 0
    for chunk in special_tokens_bytes:
        vocab[cur_id] = chunk
        cur_id += 1

    for i in range(256):
        vocab[cur_id] = bytes([i])
        cur_id += 1
    return vocab

def filter_special_tokens(
    special_tokens: list[str],
    chunk: str
) -> list[str]:

    if not special_tokens:
        return [chunk]

    pattern = "|".join(re.escape(special_token) for special_token in special_tokens)

    parts = re.split(pattern, chunk)
    return parts

def add_counts_for_filtered_chunk(
        chunk: str,
        counts: Counter[tuple[bytes, ...]],
) -> None:
    for match in re.finditer(PAT, chunk):
        token_str: str = match.group()  # newest, a string
        token_bytes = token_str.encode("utf-8")  # b"newest"

        # (b"n", b"e", b"w", b"e", b"s", b"t"), byte tuple
        token_tuple = tuple(bytes([b]) for b in token_bytes)
        counts[token_tuple] += 1

def add_pair_count_for_token_counts(
        pair_counts: Counter[tuple[bytes, bytes]],
        token_counts: Counter[tuple[bytes, ...]]
) -> None:
    for token_tuple, count in token_counts.items():
        for i in range(0, len(token_tuple) - 1):
            pair_tuple = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair_tuple] += count

def choose_max_pair_count(
        pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    return max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

def merge_pair_in_token_counts(
        token_counts: Counter[tuple[bytes, ...]],
        pair_to_merge: tuple[bytes, bytes]
) -> Counter[tuple[bytes, ...]]:
    new_token_counts: Counter[tuple[bytes, ...]] = Counter()

    left_bytes = pair_to_merge[0]
    right_bytes = pair_to_merge[1]

    for token_tuple, count in token_counts.items():
        new_token: list[bytes] = []

        token_idx = 0
        while token_idx < len(token_tuple):
            if token_idx < len(token_tuple) - 1 and token_tuple[token_idx] == left_bytes and token_tuple[token_idx + 1] == right_bytes:
                new_token.append(left_bytes + right_bytes)
                token_idx += 2
            else:
                new_token.append(token_tuple[token_idx])
                token_idx += 1
        new_token_counts[tuple(new_token)] += count
    return new_token_counts

def count_chunk_worker(job: tuple[str, int, int, list[str]]):
    input_path, start, end, special_tokens = job
    local_counts: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    filtered_chunks = filter_special_tokens(special_tokens, chunk)
    for filtered_chunk in filtered_chunks:
        add_counts_for_filtered_chunk(filtered_chunk, local_counts)

    return local_counts




def train_bpe(
        input_path: str | os.PathLike,
        max_vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]
    # dict[int, bytes]:
    # example:
    # {
    #   0: b"<|endoftext|>",
    #   1: b"a"
    # }
    # This is a vocabulary.
    
    # list[tuple[bytes, bytes]]:
    # A list of merge pairs
    # [
    #   (b"s", b"t"),
    #   (b"e", b"st")
    # ]

    merges: list[tuple[bytes, bytes]] = []
    vocab = init_vocab(special_tokens_bytes)
    counts: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, "rb") as f:
        num_chunks = 1
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens_bytes[0])

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            # Filter the special token
            filtered_chunks = filter_special_tokens(special_tokens, chunk)
            for filtered_chunk in filtered_chunks:
                # Add each token to counts
                add_counts_for_filtered_chunk(filtered_chunk, counts)
    
    while len(vocab) < max_vocab_size:
        # Process counts, get pair counts
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        add_pair_count_for_token_counts(pair_counts, counts)

        if not pair_counts:
            break
        pair_with_max_pair_count = choose_max_pair_count(pair_counts)
    
        new_vocab_bytes = pair_with_max_pair_count[0] + pair_with_max_pair_count[1]
        vocab_id = len(vocab)
        vocab[vocab_id] = new_vocab_bytes
        merges.append(pair_with_max_pair_count)
    
        # print(counts)
        # print(pair_with_max_pair_count)
    
        counts = merge_pair_in_token_counts(counts, pair_with_max_pair_count)
    
        # print("update counts", counts)

    # print("vocab size:", len(vocab))
    # print("num merges:", len(merges))
    # print("merges:", merges)
    return (vocab, merges)


if __name__ == "__main__":
    special_tokens: list[str] = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        "data/bpe_example.txt",
        263,
        special_tokens)
    # print(vocab)
    # print(merges)
