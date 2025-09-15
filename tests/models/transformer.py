from dltoolkit.models.transformer.tokenizer import BPETokenizer, BPETokenizerTrainer


def test_bpe_tokenizer():
    # test BPE Tokenizer Trainer
    input_path = 'tests/data/corpus.en'
    special_tokens = ["<|endoftext|>"]
    trainer = BPETokenizerTrainer(special_tokens=special_tokens)
    trainer.train(input_path, vocab_size=500)
    vocab = trainer.get_vocab()
    merges = trainer.get_merges()

    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded = tokenizer.encode(text)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)