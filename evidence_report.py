import matplotlib.pyplot as plt
import io, base64

def plot_token_prob_trend(token_log_probs, outlier_indices=None):
    """Return PNG as data URI showing token log-prob trend."""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(token_log_probs, marker='.', linestyle='-')
    if outlier_indices:
        ax.scatter(outlier_indices, [token_log_probs[i] for i in outlier_indices], color='red')
    ax.set_xlabel('token index')
    ax.set_ylabel('log-prob')
    ax.grid(True)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{data}"

def html_highlight_tokens(tokens, outlier_indices):
    """Return HTML snippet with outlier tokens highlighted."""
    fragments = []
    for i, t in enumerate(tokens):
        safe = t.replace('<', '&lt;').replace('>', '&gt;')
        if i in outlier_indices:
            fragments.append(f"<mark style='background-color: #FFFF33; color: black;'>{safe}</mark>")
        else:
            fragments.append(safe)
    return '<span style="font-family: monospace;">' + ' '.join(fragments) + '</span>'
