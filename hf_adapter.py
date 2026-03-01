from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HFAdapter:
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if 'cuda' in self.device else None
        ).to(self.device)
        self.model.eval()

    def token_log_probs(self, text):
        toks = self.tokenizer(text, return_tensors='pt')
        input_ids = toks['input_ids'].to(self.device)
        attention_mask = toks['attention_mask'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = []
        token_strs = []
        for i in range(input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            if i == 0:
                token_log_probs.append(float(log_probs[0, i, token_id]))
            else:
                token_log_probs.append(float(log_probs[0, i-1, token_id]))
            token_strs.append(self.tokenizer.decode([token_id]))
        return token_strs, token_log_probs

    def chunked_token_log_probs(self, text, chunk_size=50):
        """Split text into chunks of approx chunk_size tokens and get avg min-k% prob per chunk."""
        toks = self.tokenizer(text)
        input_ids = toks['input_ids']
        # Convert input ids to tokens strings (optional)
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids]

        # Break input ids into chunks of chunk_size
        chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]

        chunk_scores = []
        chunk_indices = []

        # Score each chunk separately
        for idx, chunk_ids in enumerate(chunks):
            chunk_text = self.tokenizer.decode(chunk_ids)
            _, log_probs = self.token_log_probs(chunk_text)
            # Use existing min_k_percent_prob for bottom 20% tokens in chunk
            from min_k_prob import min_k_percent_prob
            score = min_k_percent_prob(log_probs, k_percent=0.5)
            chunk_scores.append(score)
            chunk_indices.append(idx)

        # Aggregate: max score among chunks
        max_score = max(chunk_scores) if chunk_scores else float('-inf')

        return max_score, chunk_scores, chunk_indices