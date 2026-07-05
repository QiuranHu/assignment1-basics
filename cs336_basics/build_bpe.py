import os

from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
import multiprocessing as mp


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


'''
Example:
token_tuple: (b"n", b"e", b"w")

output:
[(b"n", b"e"), (b"e", b"w")]
'''
def get_pairs(token_tuple: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    pairs: list[tuple[bytes, bytes]] = []
    for i in range(len(token_tuple) - 1):
        pairs.append((token_tuple[i], token_tuple[i + 1]))

    return pairs


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

"""
token_tuple = (b"n", b"e", b"w")
count = 3

it has pairs
(b"n", b"e")
(b"e", b"w")

When BPE chooses a pair to merge, only token tuples that contain that pair will change. 
For example, if token_tuple = (b"n", b"e", b"w") appears 3 times and the chosen merge is (b"n", b"e"), 
then the old tuple (b"n", b"e", b"w") is replaced by the new tuple (b"ne", b"w"). 
That means the old tuple’s pair information is no longer true: before the merge,
 it contributed +3 to (b"n", b"e") and +3 to (b"e", b"w"); after the merge, 
 those old pairs disappear and the new pair (b"ne", b"w") appears instead. 
 So remove_token_from_caches subtracts the old tuple’s contribution from pair_counts and removes 
 the old tuple from pair_to_tokens. Then another helper adds the new tuple’s contribution back. 
 
 In short: remove old pair info, create merged token tuple, add new pair info.
"""
def remove_token_from_caches(
        token_tuple: tuple[bytes, ...], # old token to remove, after one merge, old token is gone
        count: int, # how many times token appeared
        pair_counts: Counter[tuple[bytes, bytes]],
        pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    pairs = get_pairs(token_tuple)
    for pair in pairs:
        pair_counts[pair] -= count

        if pair_counts[pair] <= 0:
            del pair_counts[pair]

    for pair in set(pairs):
        if pair in pair_to_tokens:
            pair_to_tokens[pair].discard(token_tuple)

            if len(pair_to_tokens[pair]) == 0:
                del pair_to_tokens[pair]

def add_token_to_caches(
        token_tuple: tuple[bytes, ...],
        count: int,
        pair_counts: Counter[tuple[bytes, bytes]],
        pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    pairs = get_pairs(token_tuple)

    for pair in pairs:
        pair_counts[pair] += count

    for pair in set(pairs):
        pair_to_tokens[pair].add(token_tuple)

"""
This function takes one token tuple and one pair to merge, 
then returns a new token tuple where every non-overlapping occurrence of that pair
 has been combined into one bigger byte token.
  
  For example, if the token tuple is (b"n", b"e", b"w") and the pair to merge is (b"n", b"e"),
   the function returns (b"ne", b"w"). It does not change counts, vocab, or merges by itself; 
   it only answers: “what does this one token tuple look like after applying this merge?
”"""
def merge_pair_in_token_tuple(
        token_tuple: tuple[bytes, ...],
        pair_to_merge: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    new_token: list[bytes] = []
    left_bytes = pair_to_merge[0]
    right_bytes = pair_to_merge[1]

    token_idx = 0
    while token_idx < len(token_tuple):
        if (
            token_idx < len(token_tuple) - 1
            and token_tuple[token_idx] == left_bytes
            and token_tuple[token_idx + 1] == right_bytes
        ):
            new_token.append(left_bytes + right_bytes)
            token_idx += 2
        else:
            new_token.append(token_tuple[token_idx])
            token_idx += 1
    return tuple(new_token)

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
    """
    counts = Counter({
        (b"n", b"e", b"w"): 2,
        (b"l", b"o", b"w"): 1,
    })
    """

    with open(input_path, "rb") as f:
        num_chunks = os.cpu_count() or 1
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens_bytes[0])

    jobs: list[tuple[str, int, int, list[str]]] = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with mp.Pool(processes=num_chunks) as pool:
        chunk_counts_list = pool.map(count_chunk_worker, jobs)

    for chunk_counts in chunk_counts_list:
        counts.update(chunk_counts)

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    # How many times does the pair appear
    """
    pair_counts = {
        (b"n", b"e"): 3,
        (b"e", b"w"): 3,
        (b"l", b"o"): 2,
    }
    """
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    # Which token tuples contain this pair
    """
    pair_to_tokens = {
        (b"n", b"e"): {(b"n", b"e", b"w")},
        (b"l", b"o"): {(b"l", b"o", b"w")},
    }
    """

    for token_tuple, count in counts.items():
        add_token_to_caches(token_tuple, count, pair_counts, pair_to_tokens)

    while len(vocab) < max_vocab_size:
        if not pair_counts:
            break
        pair_with_max_pair_count = choose_max_pair_count(pair_counts)
    
        new_vocab_bytes = pair_with_max_pair_count[0] + pair_with_max_pair_count[1]
        vocab_id = len(vocab)
        vocab[vocab_id] = new_vocab_bytes
        merges.append(pair_with_max_pair_count)
    
        # print(counts)
        # print(pair_with_max_pair_count)
        affected_tokens = list(pair_to_tokens[pair_with_max_pair_count])

        for old_token_tuple in affected_tokens:
            if old_token_tuple not in counts:
                continue
            count = counts[old_token_tuple]
            remove_token_from_caches(old_token_tuple, count, pair_counts, pair_to_tokens)

            del counts[old_token_tuple]

            new_token_tuple = merge_pair_in_token_tuple(old_token_tuple, pair_with_max_pair_count)
            counts[new_token_tuple] += count

            add_token_to_caches(new_token_tuple, count, pair_counts, pair_to_tokens)
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
