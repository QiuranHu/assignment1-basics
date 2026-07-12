import time

from cs336_basics.build_bpe import train_bpe
import pickle

vocab_size = 10000
special_tokens = ["<|endoftext|>"]

def main():
    start_time = time.time()

    vocab, merges = train_bpe(
        # input_path="data/TinyStoriesV2-GPT4-valid.txt",
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        max_vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    print("vocab size", len(vocab))
    print("merges", len(merges))

    longest_token: bytes = max(vocab.values(), key=lambda token : len(token))

    print("longest token bytes:", longest_token)
    print("longest token length:", len(longest_token))
    print("longest token decoded:", longest_token.decode("utf-8", errors="replace"))


    end_time = time.time()
    print("time seconds", end_time - start_time)

    with open("tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

if __name__ == "__main__":
    main()
