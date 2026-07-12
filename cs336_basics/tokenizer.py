import pickle
from collections.abc import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_pair(
        pieces: tuple[bytes, ...],
        pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    # scan left to right
    # when 2 neighbor pieces equal pair
    #   combine them
    #   move forward by 2
    # otherwize:
    #   keep current piece
    #   move forward by 1
    merged: list[bytes] = []
    i = 0
    while i < len(pieces):
        if (
            i < len(pieces) - 1
            and pieces[i] == pair[0]
            and pieces[i + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(pieces[i])
            i += 1
    return tuple(merged)

def find_best_pair(
        pieces: tuple[bytes, ...],
        merge_ranks: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes] | None:
    best_pair = None
    best_rank = None

    # Loop through neighboring pieces
    # make pair = (pieces[i], pieces[i + 1])
    # if pair exists in merge_ranks:
    #   compare its rank with best_rank
    #   keep the pair with the smallest rank

    for i in range(len(pieces) - 1):
        pair = (pieces[i], pieces[i + 1])
        if pair in merge_ranks:
            rank = merge_ranks[pair]

            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_pair = pair

    return best_pair

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }
        self.special_tokens= special_tokens or []

        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

        next_id = max(self.vocab.keys()) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                self.vocab[next_id] = token_bytes
                self.token_to_id[token_bytes] = next_id
                next_id += 1


    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(
                re.escape(token) for token in sorted_tokens
            ) + ")"
            parts = re.split(pattern=special_pattern, string=text)
        else:
            parts = [text]
        
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                token_ids.append(
                    self.token_to_id[part.encode("utf-8")]
                )
                continue

            for match in re.finditer(PAT, part):
                pre_token = match.group()
                pre_token_bytes = pre_token.encode("utf-8")
                pieces = tuple(bytes([b]) for b in pre_token_bytes)

                while True:
                    best_pair = find_best_pair(pieces, self.merge_ranks)
                    if best_pair is None:
                        break
                    pieces = merge_pair(pieces, best_pair)

                for piece in pieces:
                    token_ids.append(self.token_to_id[piece])

                # print(pieces)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)




if __name__ == "__main__":
    vocab = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
    }

    merges = [
        (b"t", b"h"),
        (b" ", b"c"),
        (b" ", b"a"),
        (b"th", b"e"),
        (b" a", b"t"),
    ]

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=None,
    )

    token_ids = tokenizer.encode("the cat ate")
    print(token_ids)

    print(tokenizer.decode(token_ids))
