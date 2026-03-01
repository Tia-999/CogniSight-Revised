from hf_adapter import HFAdapter

adapter = HFAdapter("gpt2")
text = "Hello, this is a test with GPT-2."
tokens, log_probs = adapter.token_log_probs(text)

print("Tokens:", tokens)
print("Log probabilities:", log_probs)
