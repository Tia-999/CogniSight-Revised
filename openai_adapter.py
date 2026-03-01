import os
import openai

class OpenAIAdapter:
    def __init__(self, api_key=None, engine='text-davinci-003'):
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        openai.api_key = api_key
        self.engine = engine

    def token_log_probs(self, text):
        resp = openai.Completion.create(
            model=self.engine,
            prompt=text,
            max_tokens=0,
            echo=True,
            logprobs=0
        )
        lp = resp['choices'][0]['logprobs']['token_logprobs']
        toks = resp['choices'][0]['logprobs']['tokens']
        return toks, lp
