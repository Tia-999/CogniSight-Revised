from datasets import load_dataset
from hf_adapter import HFAdapter
from min_k_prob import min_k_percent_prob
from calibration import calibrate_threshold
from sklearn.metrics import precision_recall_fscore_support
import argparse

def evaluate_detection(model_name, dataset_split='WikiMIA_length128', k_percent=0.2, target_fpr=0.05):
    ds = load_dataset('swj0419/WikiMIA', split=dataset_split)
    adapter = HFAdapter(model_name)
    
    texts = [ex.get('input') or ex.get('content') or '' for ex in ds]
    labels = [ex['label'] for ex in ds]

    scores = []
    print(f"Evaluating {len(ds)} samples with model '{model_name}' ...")
    for i, text in enumerate(texts):
        toks, lps = adapter.token_log_probs(text)
        score = min_k_percent_prob(lps, k_percent=k_percent)
        scores.append(score)
        if (i+1) % 50 == 0:
            print(f"Processed {i+1} / {len(ds)}")

    thr, tpr = calibrate_threshold(scores, labels, target_fpr=target_fpr)
    print(f"\nThreshold for target FPR={target_fpr*100:.1f}%: {thr:.4f}")
    print(f"True positive rate (recall) at threshold: {tpr:.4f}")

    preds = [1 if s <= thr else 0 for s in scores]
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate detection score on WikiMIA dataset")
    parser.add_argument('--model', type=str, required=True, help='HF model name (e.g. EleutherAI/pythia-2.8b)')
    parser.add_argument('--split', type=str, default='WikiMIA_length128', help='Dataset split name')
    parser.add_argument('--kpercent', type=float, default=0.2, help='k% of lowest token log-probs to average')
    parser.add_argument('--fpr', type=float, default=0.05, help='Target false positive rate for threshold calibration')
    args = parser.parse_args()

    evaluate_detection(args.model, args.split, args.kpercent, args.fpr)