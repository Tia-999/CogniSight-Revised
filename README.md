# ⚡ Cognisight

**Shine a light on what your LLM remembers**

> “Was this text in the model’s training set?”
> Cognisight gives you a **fast, black-box signal** using **MIN-K% + adaptive outlier detection**, with visual highlights and token-probability traces that anyone can interpret at a glance.

On GPT-2 with 128-token chunks, Cognisight reliably reaches **\~0.68–0.69 AUC** on WikiMIA mini-runs — balancing accuracy and compute with **z-score adaptive mode**.

---

## 🏅 Badges

* ✅ Made with **Streamlit**
* 🔒 Black-box friendly (**no weights required**)
* 📈 Repro-check: AUC ≈ 0.68 on 100 WikiMIA samples (len=128)

---

## 🔍 What Cognisight Does

* **Token-level evidence**

  * Computes token log-probabilities
  * Flags the bottom **K% “surprise tokens”** via MIN-K%
  * Aggregates into a single familiarity score

* **Adaptive modes**

  * **z-score**: relative deviation from text mean
  * **IQR**: stable detection with resistant quartile thresholds
  * Both normalize across styles, lengths, and vocab distributions

* **Transparent visuals**

  * Highlighted tokens show “what went wrong”
  * Probability trend plots reveal dips & spikes where the model hesitated

* **One-click sanity checks**

  * Built-in 100-sample **mini-benchmark** on WikiMIA
  * Get a quick ROC-AUC curve for member vs. non-member separation

---

## 💡 Why It Matters

* 🛡 **Copyright & privacy audits** — flag passages that look memorized
* 📚 **Contamination checks** — ensure evaluation text wasn’t seen during pretraining
* ⚙️ **Model-agnostic** — works with Hugging Face models or OpenAI APIs (when logprobs available)

---

## ⚡ Quick Start

### 1. Install

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run streamlit_App.py
```

### 3. Use the UI

* **Sidebar**

  * Choose backend: `HuggingFace` or `OpenAI`
  * Input model name (e.g., `gpt2`)
  * Toggle **adaptive (z-score)** or switch to **IQR**

* **Main panel**

  * Paste text → click **Run detection**
  * See detection score, highlighted tokens, and probability plot
  * Click **Run AUC Evaluation** for a mini-benchmark on `WikiMIA_length128`

---

## 📊 Expected Results

* **Demo profile (WikiMIA, 100 samples):**

  * MIN-K% (len=128): **0.6842 AUC**
  * Adaptive z-score (len=128): **0.6752 AUC**
  * MIN-K% (len=64): **0.6319 AUC**

* **Trends:**

  * Longer chunks → stronger separation
  * **z-score**: robust & efficient
  * **IQR**: conservative, may underperform on some splits

---

## ⚙️ How It Works (1-Minute Version)

1. **MIN-K%**

   * Get log-probability for each token
   * Select bottom *k%* (most “surprising”)
   * Average them → familiarity score

2. **Adaptive normalizers**

   * **z-score**: how many σ below average is this token?
   * **IQR**: uses quartile spread to filter strong outliers

3. **Evidence UI**

   * Marked tokens = the surprises
   * Plot shows where the model stumbled

---

## 🗂 Repo Map

* `hf_adapter.py` — Hugging Face interface (log-probs, chunked scoring)
* `openai_adapter.py` — OpenAI completions with log-probs (optional)
* `min_k_prob.py` — Bottom-K% scoring logic
* `adaptive_detector.py` — z-score/IQR adaptive detection
* `evidence_report.py` — HTML highlighting + trend plots
* `eval.py` — Batch scoring, calibrated thresholds, metrics
* `streamlit_App.py` — Interactive dashboard (text + AUC mode)
* `cli.py` — Lightweight CLI runner

---

## 🧪 Try These Examples

* **Familiar text (older Wikipedia)**

  * Expect higher (less negative) scores, few highlights
* **Unfamiliar text (recent/new topics)**

  * Expect lower scores, many red highlights, sharp probability dips

---

## 🖥 CLI Snippets

Quick pass over 20 dataset items:

```bash
python cli.py --model gpt2
```

Evaluate with custom settings:

```bash
python eval.py --model gpt2 --kpercent 0.2 --fpr 0.05
```

---

## 💡 Tips & Best Practices

* **Chunk length matters** → prefer 128 tokens for stronger signals
* **Use z-score mode** for balanced speed/accuracy
* **IQR mode** is useful with noisy or spiky distributions
* **Explainability is key** → use highlights + plots to show *why* a score was assigned

---

## 📜 License

MIT License. See `LICENSE` for details.

---
