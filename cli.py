import argparse
from datasets import load_dataset
from hf_adapter import HFAdapter
from min_k_prob import min_k_percent_prob
from adaptive_detector import adaptive_min_prob
from evidence_report import html_highlight_tokens, plot_token_prob_trend

def run_on_dataset(model_name, use_adaptive=False, method='zscore'):
    ds = load_dataset('swj0419/WikiMIA', split='WikiMIA_length128')
    adapter = HFAdapter(model_name)
    results = []
    for ex in ds.select(range(20)):
        text = ex.get('input') or ''
        toks, lps = adapter.token_log_probs(text)
        if use_adaptive:
            avg, idx = adaptive_min_prob(lps, method=method)
        else:
            avg = min_k_percent_prob(lps, k_percent=0.5)
            idx = sorted(range(len(lps)), key=lambda j: lps[j])[:max(1, int(0.2*len(lps)))]
        html = html_highlight_tokens(toks, idx)
        img = plot_token_prob_trend(lps, idx)
        results.append({'avg': avg, 'html': html, 'plot': img})
    return results


def run_on_dataset_chunked(model_name):
    ds = load_dataset('swj0419/WikiMIA', split='WikiMIA_length128')
    adapter = HFAdapter(model_name)
    results = []
    for ex in ds.select(range(10)):
        text = ex.get('input') or ''
        max_score, chunk_scores, _ = adapter.chunked_token_log_probs(text, chunk_size=50)
        results.append({'max_chunk_score': max_score})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--adaptive', action='store_true')
    args = parser.parse_args()
    res = run_on_dataset(args.model, use_adaptive=args.adaptive)
    print('done', len(res))
