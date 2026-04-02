import csv
import streamlit as st
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EXP_ROOT = BASE_DIR.parent / "political-bias-representation-engineering"
STEP1_DIR = EXP_ROOT / "step1_politicality_v2"
OLD_STEP1_DIR = EXP_ROOT / "results_step1"
STEP1_LIFESTYLE = STEP1_DIR / "outputs_huffpost_lifestyle_random"
STEP1_STRICT = STEP1_DIR / "outputs_huffpost_random_strict_headline_only"
STEP1_BBC = STEP1_DIR / "outputs_bbc_random"
STEP1_POL_OR_NOT = STEP1_DIR / "outputs"
STEP1_HUFFPOST_ORIG = STEP1_DIR / "outputs_huffpost_random"
OLD_STEP1_COMBINED_COSINE = EXP_ROOT / "results_combined_step1_cosine.png"
STEP2_DIR = EXP_ROOT / "step2_ideology_public_v1"
STEP2_COMPARISON = STEP2_DIR / "comparison"
STEP2_ALLSIDES = STEP2_DIR / "comparison_allsides"


def load_preview_sections(preview_path: Path, limit: int = 4) -> tuple[list[str], list[str]]:
    if not preview_path.exists():
        return [], []

    political: list[str] = []
    non_political: list[str] = []
    current: str | None = None
    for raw_line in preview_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("===="):
            continue
        if line.startswith("Political preview"):
            current = "political"
            continue
        if line.startswith("Non-political preview"):
            current = "non_political"
            continue
        if not line.startswith("["):
            continue
        if current == "political" and len(political) < limit:
            political.append(line)
        elif current == "non_political" and len(non_political) < limit:
            non_political.append(line)
    return political, non_political


def peak_gap_markdown(summary_csv: Path) -> str:
    if not summary_csv.exists():
        return "_Summary file not found._"

    rows = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    row["model_name"],
                    float(row["peak_angular_gap"]),
                    int(row["peak_layer"]),
                )
            )

    lines = ["| Model | Peak angular gap | Peak layer |", "|---|---:|---:|"]
    for model_name, peak_gap, peak_layer in rows:
        lines.append(f"| {model_name} | {peak_gap:.3f}° | {peak_layer} |")
    return "\n".join(lines)

def render_preview_examples(preview_path: Path):
    political, non_political = load_preview_sections(preview_path)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Political examples**")
        if political:
            for item in political:
                st.markdown(f"- {item}")
        else:
            st.caption("No preview found")
    with col2:
        st.markdown("**Non-political examples**")
        if non_political:
            for item in non_political:
                st.markdown(f"- {item}")
        else:
            st.caption("No preview found")


def render_custom_matched_examples():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Left examples**")
        st.markdown("- [healthcare] Healthcare is a fundamental human right...")
        st.markdown("- [gun_control] We need stricter gun control laws...")
        st.markdown("- [immigration] Immigrants strengthen our economy...")
    with col2:
        st.markdown("**Right examples**")
        st.markdown("- [healthcare] Healthcare should be driven by free market competition...")
        st.markdown("- [gun_control] The Second Amendment guarantees the individual right...")
        st.markdown("- [immigration] We must secure our borders and enforce existing laws...")
    with col3:
        st.markdown("**Neutral examples**")
        st.markdown("- [healthcare] The U.S. healthcare system involves a mix of private insurance and government programs...")
        st.markdown("- [gun_control] Gun policy in the United States is shaped by the Second Amendment, state laws, and federal regulations...")
        st.markdown("- [immigration] The United States admits roughly one million legal immigrants per year...")


def load_threeway_preview_sections(preview_path: Path, limit: int = 4) -> tuple[list[str], list[str], list[str]]:
    if not preview_path.exists():
        return [], [], []

    left: list[str] = []
    center: list[str] = []
    right: list[str] = []
    current: str | None = None

    for raw_line in preview_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "## LEFT":
            current = "left"
            continue
        if line == "## CENTER":
            current = "center"
            continue
        if line == "## RIGHT":
            current = "right"
            continue
        if not line.startswith("- title:"):
            continue
        if current == "left" and len(left) < limit:
            left.append(line.replace("- title: ", "", 1))
        elif current == "center" and len(center) < limit:
            center.append(line.replace("- title: ", "", 1))
        elif current == "right" and len(right) < limit:
            right.append(line.replace("- title: ", "", 1))

    return left, center, right


def render_threeway_preview_examples(preview_path: Path) -> None:
    left, center, right = load_threeway_preview_sections(preview_path)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Left examples**")
        if left:
            for item in left:
                st.markdown(f"- {item}")
        else:
            st.caption("No preview found")
    with col2:
        st.markdown("**Center examples**")
        if center:
            for item in center:
                st.markdown(f"- {item}")
        else:
            st.caption("No preview found")
    with col3:
        st.markdown("**Right examples**")
        if right:
            for item in right:
                st.markdown(f"- {item}")
        else:
            st.caption("No preview found")


def render_allsides_preview(preview_path: Path, limit: int = 2) -> None:
    if not preview_path.exists():
        st.caption("No preview found")
        return
    blocks = []
    current = []
    for raw_line in preview_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw_line.startswith("## Event:") and current:
            blocks.append("\n".join(current).strip())
            current = [raw_line]
        else:
            current.append(raw_line)
    if current:
        blocks.append("\n".join(current).strip())

    for block in blocks[:limit]:
        st.code(block, language="text")


def step2_probe_markdown(summary_csv: Path) -> str:
    if not summary_csv.exists():
        return "_Summary file not found._"

    rows = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    row["model_name"],
                    float(row["peak_probe_auroc"]),
                    int(row["peak_probe_auroc_layer"]),
                    float(row["candidate_region_mean_probe_auroc"]),
                )
            )

    lines = ["| Model | Peak probe AUROC | Peak layer | Candidate-region mean AUROC |", "|---|---:|---:|---:|"]
    for model_name, peak_auc, peak_layer, region_auc in rows:
        lines.append(f"| {model_name} | {peak_auc:.3f} | {peak_layer} | {region_auc:.3f} |")
    return "\n".join(lines)


def step2_compare_markdown(compare_csv: Path) -> str:
    if not compare_csv.exists():
        return "_Comparison file not found._"

    rows = []
    with compare_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    row["model_name"],
                    float(row["article_bias_peak_probe_auroc"]),
                    float(row["allsides_peak_probe_auroc"]),
                    row["better_probe_dataset"],
                )
            )

    lines = ["| Model | Article-Bias peak AUROC | AllSides peak AUROC | Better dataset |", "|---|---:|---:|---|"]
    for model_name, article_auc, allsides_auc, better in rows:
        lines.append(f"| {model_name} | {article_auc:.3f} | {allsides_auc:.3f} | {better} |")
    return "\n".join(lines)


def render_step2_figure_guide() -> None:
    st.markdown(
        """
**What these figures mean**
- **3-class probe**
  asks: **can this layer decode `left / center / right` at all?**
  If the probe AUROC is high, ideology labels are more linearly readable from that layer.

- **Bias-direction geometry**
  asks: **is `center` geometrically balanced between left and right, or closer to one side?**
  Values near zero mean a more symmetric geometry.
  Positive or negative shifts mean `center` is not equally placed between the two sides.

- **Center projection**
  asks: **where does `center` actually fall on the left-right axis?**
  Near the middle means it behaves like a true middle anchor.
  A drift toward one side means the "center" representation is not geometrically centered.

So the three plots answer different questions:
- probe = **can we decode the three labels?**
- bias direction = **is the geometry symmetric?**
- center projection = **where is the middle?**
"""
    )


def plot_cosine_angle():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    fig, ax = plt.subplots(figsize=(9.2, 3.4), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    origin = (0.7, 0.6)
    v1 = (4.2, 1.4)
    v2 = (3.1, 2.7)

    ax.annotate("", xy=(5.6, origin[1]), xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 2.4})
    ax.annotate("", xy=(origin[0], 3.2), xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 2.4})
    ax.text(5.65, origin[1] - 0.06, "x", fontsize=13)
    ax.text(origin[0] - 0.1, 3.25, "y", fontsize=13)

    ax.annotate("", xy=v1, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": "#2563eb"})
    ax.annotate("", xy=v2, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": "#f97316"})
    ax.text(v1[0] + 0.12, v1[1] - 0.12, "v(P)", color="#2563eb", fontsize=13, weight="bold")
    ax.text(v2[0] + 0.12, v2[1] + 0.06, "v(NP)", color="#f97316", fontsize=13, weight="bold")

    arc = Arc(origin, width=2.2, height=2.2, angle=0, theta1=10, theta2=44, lw=2.4)
    ax.add_patch(arc)
    ax.text(origin[0] + 1.25, origin[1] + 0.72, "θ", fontsize=13)

    ax.text(0.7, 3.45, "Cosine similarity as an angle", fontsize=16, weight="bold")
    box = dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC", edgecolor="#E5E7EB", linewidth=1.2)
    ax.text(
        4.05,
        3.05,
        "Cosine -> angle\nsame direction = small angle",
        fontsize=13,
        bbox=box,
        va="top",
    )
    ax.text(4.1, 2.1, "Small θ → similar", fontsize=13)
    ax.text(4.1, 1.78, "Large θ → dissimilar", fontsize=13)

    st.pyplot(fig, use_container_width=True)


def pipeline_dot() -> str:
    return r"""
digraph G {
  rankdir=TB;
  bgcolor="white";
  graph [fontname="Helvetica", fontsize=14, pad="0.2", nodesep="0.35", ranksep="0.45"];
  node  [fontname="Helvetica", fontsize=14, shape=box, style="rounded,filled", color="#CBD5E1", fillcolor="#F8FAFC", margin="0.16,0.10"];
  edge  [color="#111827", arrowsize=0.8];

  A [label="Inputs\nP statements + NP statements"];
  B [label="Wrap with prompt template"];
  C [label="Forward pass (per layer)\nExtract last-token hidden state h[l]"];
  D [label="Random pairing (R rounds, per layer)\nP–P, NP–NP, P–NP"];
  E [label="Cosine similarity\ncos = (x·y)/(||x|| ||y||)"];
  F [label="Angle conversion\nθ = arccos(clip(cos, -1, 1))"];
  G2 [label="Angular gap per layer\n gap[l] = mean(θ_PN[l] − (θ_PP[l]+θ_NN[l])/2)"];

  A -> B -> C -> D -> E -> F -> G2;
}
"""


def plot_pairing_contrast():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    import math

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True)
    titles = ["Within-class: P–P (small θ)", "Within-class: NP–NP (small θ)", "Cross-class: P–NP (large θ)"]
    specs = [
        {"colors": ("#2563eb", "#2563eb"), "deg": (12, 22)},
        {"colors": ("#f97316", "#f97316"), "deg": (12, 24)},
        {"colors": ("#2563eb", "#f97316"), "deg": (12, 64)},
    ]

    for ax, title, spec in zip(axes, titles, specs):
        ax.set_aspect("equal")
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0, 3.5)
        ax.axis("off")
        origin = (0.55, 0.35)
        length = 3.0
        deg1, deg2 = spec["deg"]
        c1, c2 = spec["colors"]

        rad1 = deg1 * math.pi / 180.0
        rad2 = deg2 * math.pi / 180.0
        v1 = (origin[0] + length * math.cos(rad1), origin[1] + length * math.sin(rad1))
        v2 = (origin[0] + length * math.cos(rad2), origin[1] + length * math.sin(rad2))

        ax.annotate("", xy=v1, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": c1})
        ax.annotate("", xy=v2, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": c2})

        theta1 = min(deg1, deg2)
        theta2 = max(deg1, deg2)
        radius = 1.35
        arc = Arc(origin, width=2 * radius, height=2 * radius, angle=0, theta1=theta1, theta2=theta2, lw=2.2, color="#111827")
        ax.add_patch(arc)

        mid = (theta1 + theta2) / 2.0
        mid_rad = mid * math.pi / 180.0
        tx = origin[0] + (radius + 0.25) * math.cos(mid_rad)
        ty = origin[1] + (radius + 0.25) * math.sin(mid_rad)
        ax.text(tx, ty, "θ", fontsize=12, ha="center", va="center")

        ax.set_title(title, fontsize=12, fontweight="bold")
    st.pyplot(fig, use_container_width=True)


def plot_gap_curve_example():
    import matplotlib.pyplot as plt
    import numpy as np

    layers = np.arange(1, 33)
    peak = 12
    gap = 4 + 18 * np.exp(-((layers - peak) ** 2) / (2 * 18)) + 1.2 * np.exp(-((layers - 28) ** 2) / (2 * 10))
    fig, ax = plt.subplots(figsize=(9.6, 2.8), constrained_layout=True)
    ax.plot(layers, gap, color="#111827", lw=2.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Angular gap (°)")
    ax.set_title("Illustration: gap(l) peaks where political vs non-political separates", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.axvline(peak, color="#ef4444", lw=1.6, linestyle="--")
    ax.text(peak + 0.4, float(gap.max()) - 1.0, "peak", color="#ef4444", fontsize=11)
    st.pyplot(fig, use_container_width=True)


def plot_sigmoid_curve():
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-8, 8, 400)
    s = 1 / (1 + np.exp(-x))
    fig, ax = plt.subplots(figsize=(9.6, 2.6), constrained_layout=True)
    ax.plot(x, s, color="#2563eb", lw=2.2)
    ax.axvline(0, color="#9ca3af", ls="--", lw=1.2)
    ax.axhline(0.5, color="#9ca3af", ls="--", lw=1.2)
    ax.set_xlabel("z = w·x + b")
    ax.set_ylabel("σ(z)")
    ax.set_title("Logistic regression: sigmoid maps log-odds to probability", fontweight="bold")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def plot_logistic_regression_concept():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(7)
    pol = rng.normal(loc=(1.8, 2.2), scale=(0.22, 0.28), size=(24, 2))
    non = rng.normal(loc=(3.1, 1.2), scale=(0.28, 0.24), size=(24, 2))

    fig, ax = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
    ax.scatter(pol[:, 0], pol[:, 1], color="#2563eb", s=34, label="political")
    ax.scatter(non[:, 0], non[:, 1], color="#f97316", s=34, label="non-political")

    x = np.linspace(0.9, 4.0, 100)
    y = -0.95 * x + 4.35
    ax.plot(x, y, linestyle="--", color="#111827", lw=2, label="linear boundary")
    ax.text(2.6, 2.8, "simple linear split", fontsize=11, color="#111827")

    ax.set_title("Layer-wise logistic regression: can one simple boundary separate the classes?", fontweight="bold")
    ax.set_xlabel("hidden feature 1")
    ax.set_ylabel("hidden feature 2")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig, use_container_width=True)


def plot_probe_input_representation():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.4, 3.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F8FAFC", ec="#CBD5E1", weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.16,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#111827"})

    box(0.4, 3.7, 2.2, 1.0, "Political texts\nand\nNon-political texts", fc="#EFF6FF", ec="#93C5FD", weight="bold")
    box(3.0, 3.7, 2.0, 1.0, "Wrap with\nprompt template")
    box(5.5, 3.7, 1.8, 1.0, "Run the\nLLM once")
    box(7.8, 3.7, 2.2, 1.0, "Extract hidden\nstates from every layer")
    box(10.3, 3.7, 1.2, 1.0, "Pick\none layer", fc="#FEF3C7", ec="#F59E0B", weight="bold")

    arrow(2.6, 4.2, 3.0, 4.2)
    arrow(5.0, 4.2, 5.5, 4.2)
    arrow(7.3, 4.2, 7.8, 4.2)
    arrow(10.0, 4.2, 10.3, 4.2)

    box(3.2, 1.1, 5.4, 1.3, "At this one layer:\n200 texts -> 200 vectors + 200 labels", fc="#ECFDF5", ec="#86EFAC", weight="bold")
    ax.text(3.6, 0.55, "This becomes the dataset for one logistic-regression probe.", fontsize=10.5, color="#374151")
    arrow(10.9, 3.7, 8.0, 2.4)

    ax.set_title("Step 1 probe: what goes into the classifier?", fontsize=14, fontweight="bold")
    st.pyplot(fig, use_container_width=True)


def plot_probe_workflow():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)

    # Panel 1: choose one layer
    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 8)
    ax.axis("off")
    for i in range(6):
        y = 6.8 - i * 1.0
        color = "#FDE68A" if i == 3 else "#F8FAFC"
        edge = "#F59E0B" if i == 3 else "#CBD5E1"
        rect = FancyBboxPatch((1.3, y), 2.2, 0.6, boxstyle="round,pad=0.12", facecolor=color, edgecolor=edge, linewidth=1.3)
        ax.add_patch(rect)
        label = "layer l" if i == 3 else f"layer {i+1}"
        ax.text(2.4, y + 0.3, label, ha="center", va="center", fontsize=11, weight="bold" if i == 3 else "normal")
    ax.annotate("", xy=(3.9, 4.1), xytext=(4.6, 4.1), arrowprops={"arrowstyle": "-|>", "lw": 1.8})
    ax.text(0.2, 7.5, "1) Take hidden states\nfrom one layer", fontsize=12, weight="bold")
    ax.text(0.35, 1.0, "Each layer is tested separately", fontsize=10, color="#374151")

    # Panel 2: 5 folds
    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.35, "2) 5-fold stratified CV", fontsize=12, weight="bold")
    colors = ["#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED"]
    for row in range(5):
        y = 4.4 - row * 0.8
        for col in range(5):
            fc = colors[col] if col == row else "#E5E7EB"
            ec = colors[col] if col == row else "#CBD5E1"
            ax.add_patch(Rectangle((0.9 + col * 0.8, y), 0.55, 0.45, facecolor=fc, edgecolor=ec))
        ax.text(5.2, y + 0.22, f"fold {row+1}", va="center", fontsize=10)
    ax.text(0.9, 0.65, "colored block = test split\ngray blocks = train splits", fontsize=10, color="#374151")

    # Panel 3: aggregate profile
    ax = axes[2]
    layers = np.arange(1, 33)
    curve = 0.55 + 0.38 * np.exp(-((layers - 24) ** 2) / (2 * 18))
    ax.plot(layers, curve, color="#2563EB", lw=2.4)
    ax.axvline(24, color="#DC2626", linestyle="--", lw=1.5)
    ax.text(24.5, curve.max() - 0.03, "peak", color="#DC2626", fontsize=10)
    ax.set_title("3) Average test scores\nacross folds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC / accuracy")
    ax.grid(True, alpha=0.2)

    st.pyplot(fig, use_container_width=True)


def plot_permutation_concept():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(11)
    null_scores = rng.normal(0.56, 0.025, 200)
    observed = 0.95

    fig, ax = plt.subplots(figsize=(9.5, 3.2), constrained_layout=True)
    ax.hist(null_scores, bins=18, color="#CBD5E1", edgecolor="#94A3B8")
    ax.axvline(np.quantile(null_scores, 0.95), color="#0F766E", lw=1.8, linestyle="--", label="95% shuffled-label range")
    ax.axvline(observed, color="#DC2626", lw=2.4, label="Observed probe score")
    ax.text(observed + 0.005, ax.get_ylim()[1] * 0.76, "real result", color="#DC2626", fontsize=11)
    ax.set_title("Permutation test: compare the real probe score to shuffled-label runs", fontweight="bold")
    ax.set_xlabel("Probe confidence for the label 'political'")
    ax.set_ylabel("Count")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.15)
    st.pyplot(fig, use_container_width=True)


def _roc_curve(scores, labels):
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    pos = (labels == 1).astype(int)
    neg = (labels == 0).astype(int)

    tp = np.cumsum(pos)
    fp = np.cumsum(neg)
    tp_total = int(tp[-1]) if len(tp) else 0
    fp_total = int(fp[-1]) if len(fp) else 0

    if tp_total == 0 or fp_total == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    change = np.r_[True, scores[1:] != scores[:-1]]
    tp_at = tp[change]
    fp_at = fp[change]
    thr_at = scores[change]

    tpr = tp_at / tp_total
    fpr = fp_at / fp_total

    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    thr = np.r_[np.inf, thr_at, -np.inf]
    return fpr, tpr, thr


def _auc_trapz(fpr, tpr):
    import numpy as np

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    if hasattr(np, "trapz"):
        return float(np.trapz(tpr, fpr))
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return float(area)


def _auroc_pairwise(pos_scores, neg_scores):
    import numpy as np

    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comp = pos[:, None] - neg[None, :]
    return float((comp > 0).mean() + 0.5 * (comp == 0).mean())


def plot_accuracy_concept():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.8, 2.7), constrained_layout=True)
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    true_labels = ["P", "P", "NP", "P", "NP", "NP", "P", "NP"]
    pred_labels = ["P", "P", "P", "P", "NP", "NP", "P", "NP"]
    ax.set_title("Accuracy = exact hit rate (threshold-dependent)", fontweight="bold", pad=6)
    ax.text(0.25, 2.75, "True label", fontsize=11, weight="bold")
    ax.text(0.12, 1.58, "Probe output", fontsize=11, weight="bold")

    correct = 0
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        x = 1.25 + i * 0.95
        true_fc = "#DBEAFE" if t == "P" else "#FFEDD5"
        pred_fc = "#DCFCE7" if t == p else "#FEE2E2"
        pred_ec = "#16A34A" if t == p else "#DC2626"

        true_box = FancyBboxPatch((x - 0.28, 2.32), 0.56, 0.42, boxstyle="round,pad=0.08", facecolor=true_fc, edgecolor="#CBD5E1", linewidth=1.2)
        pred_box = FancyBboxPatch((x - 0.28, 1.14), 0.56, 0.42, boxstyle="round,pad=0.08", facecolor=pred_fc, edgecolor=pred_ec, linewidth=1.35)
        ax.add_patch(true_box)
        ax.add_patch(pred_box)
        ax.text(x, 2.53, t, fontsize=11, ha="center", va="center", weight="bold")
        ax.text(x, 1.35, p, fontsize=11, ha="center", va="center", weight="bold", color=pred_ec)
        ax.plot([x, x], [1.72, 2.18], color="#CBD5E1", lw=1.0, linestyle="--")

        mark = "✓" if t == p else "✗"
        mark_color = "#16A34A" if t == p else "#DC2626"
        ax.text(x, 0.72, mark, ha="center", va="center", fontsize=14, weight="bold", color=mark_color)
        if t == p:
            correct += 1
    ax.text(0.22, 0.15, f"Example: {correct}/{len(true_labels)} are correct -> accuracy = {correct/len(true_labels):.0%}", fontsize=10.5, color="#374151")

    st.pyplot(fig, use_container_width=True)


def plot_probe_metrics_concept():
    import matplotlib.pyplot as plt
    import numpy as np

    pos_scores = np.array([0.92, 0.88, 0.83, 0.77, 0.71])
    neg_scores = np.array([0.18, 0.27, 0.39, 0.51, 0.60])
    auroc_prob = _auroc_pairwise(pos_scores, neg_scores)
    scores = np.r_[pos_scores, neg_scores]
    labels = np.r_[np.ones_like(pos_scores), np.zeros_like(neg_scores)]
    fpr, tpr, _ = _roc_curve(scores, labels)
    auc = _auc_trapz(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.set_title("Probe scores by class (overlap ↓ ⇒ AUROC ↑)", fontweight="bold", pad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.35, 1.35)
    ax.set_yticks([1.0, 0.0], labels=["political", "non-political"])
    ax.set_xlabel("Probe score for label 'political'")
    ax.grid(True, axis="x", alpha=0.18)

    jitter = np.linspace(-0.07, 0.07, len(pos_scores))
    ax.scatter(pos_scores, 1.0 + jitter, color="#2563EB", s=70, label="political", zorder=3)
    ax.scatter(neg_scores, 0.0 + jitter, color="#F97316", s=70, label="non-political", zorder=3)
    ax.legend(frameon=True, loc="upper left")

    lo = float(max(min(pos_scores), min(neg_scores)))
    hi = float(min(max(pos_scores), max(neg_scores)))
    if lo < hi:
        ax.axvspan(lo, hi, color="#FDE68A", alpha=0.25, zorder=0)
        ax.text((lo + hi) / 2, 0.46, "overlap", fontsize=10.5, color="#92400E", weight="bold", ha="center")

    ax.text(0.02, -0.18, f"AUROC ≈ {auroc_prob:.2f}", fontsize=11, color="#111827")

    ax = axes[1]
    ax.set_title(f"ROC curve (AUC = {auc:.2f})", fontweight="bold", pad=8)
    ax.plot(fpr, tpr, color="#2563EB", lw=2.2, label="ROC")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.4, linestyle="--", label="random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True, loc="lower right")

    st.pyplot(fig, use_container_width=True)


def plot_step2_geometry_concept():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    left = (1.0, 1.0)
    center = (2.5, 2.7)
    right = (4.0, 1.0)
    ax.scatter(*left, s=120, color="#2563eb")
    ax.scatter(*center, s=120, color="#6b7280")
    ax.scatter(*right, s=120, color="#dc2626")
    ax.plot([left[0], center[0]], [left[1], center[1]], color="#2563eb", lw=2)
    ax.plot([right[0], center[0]], [right[1], center[1]], color="#dc2626", lw=2)
    ax.plot([left[0], right[0]], [left[1], right[1]], color="#111827", lw=2, linestyle="--")
    ax.text(left[0] - 0.2, left[1] - 0.35, "Left", color="#2563eb", fontsize=11)
    ax.text(center[0] - 0.28, center[1] + 0.25, "Center", color="#374151", fontsize=11)
    ax.text(right[0] - 0.15, right[1] - 0.35, "Right", color="#dc2626", fontsize=11)
    ax.set_title("3-class probing", fontweight="bold")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    left = (1.3, 1.1)
    right = (8.7, 1.1)
    center = (5.2, 2.8)
    proj = (5.2, 1.1)
    ax.annotate("", xy=right, xytext=left, arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": "#111827"})
    ax.scatter(*left, s=110, color="#2563eb")
    ax.scatter(*right, s=110, color="#dc2626")
    ax.scatter(*center, s=110, color="#6b7280")
    ax.scatter(*proj, s=80, color="#059669")
    ax.plot([center[0], proj[0]], [center[1], proj[1]], linestyle="--", color="#6b7280", lw=1.7)
    ax.text(left[0] - 0.2, left[1] - 0.4, "Left", color="#2563eb", fontsize=11)
    ax.text(right[0] - 0.25, right[1] - 0.4, "Right", color="#dc2626", fontsize=11)
    ax.text(center[0] - 0.35, center[1] + 0.22, "Center", color="#374151", fontsize=11)
    ax.text(proj[0] - 0.38, proj[1] + 0.2, "projection", color="#059669", fontsize=11)
    ax.set_title("Center projection on the left-right axis", fontweight="bold")

    st.pyplot(fig, use_container_width=True)


def plot_step2_input_representation():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.6, 3.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F8FAFC", ec="#CBD5E1", weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.16,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#111827"})

    box(0.35, 3.75, 2.3, 1.0, "Left / Center / Right\ntexts", fc="#EEF2FF", ec="#C7D2FE", weight="bold")
    box(3.0, 3.75, 1.9, 1.0, "Wrap with\nprompt template")
    box(5.35, 3.75, 1.8, 1.0, "Run the\nLLM once")
    box(7.6, 3.75, 2.1, 1.0, "Extract hidden\nstates by layer")
    box(10.05, 3.75, 1.35, 1.0, "Pick\none layer", fc="#FEF3C7", ec="#F59E0B", weight="bold")

    arrow(2.65, 4.25, 3.0, 4.25)
    arrow(4.9, 4.25, 5.35, 4.25)
    arrow(7.15, 4.25, 7.6, 4.25)
    arrow(9.7, 4.25, 10.05, 4.25)

    box(3.1, 1.05, 5.9, 1.45, "At this layer:\n300 texts -> 300 vectors + 3 labels\n(left / center / right)", fc="#ECFDF5", ec="#86EFAC", weight="bold")
    arrow(10.7, 3.75, 8.25, 2.5)
    ax.text(3.25, 0.5, "This becomes one 3-class probe dataset for Step 2.", fontsize=10.5, color="#374151")
    ax.set_title("Step 2 input: what goes into the ideology probe?", fontsize=14, fontweight="bold")
    st.pyplot(fig, use_container_width=True)


def plot_step2_probe_concept():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(5)
    left = rng.normal(loc=(1.5, 1.0), scale=(0.18, 0.18), size=(18, 2))
    center = rng.normal(loc=(2.6, 2.2), scale=(0.18, 0.18), size=(18, 2))
    right = rng.normal(loc=(3.7, 1.0), scale=(0.18, 0.18), size=(18, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(left[:, 0], left[:, 1], color="#2563EB", s=28, label="left")
    ax.scatter(center[:, 0], center[:, 1], color="#6B7280", s=28, label="center")
    ax.scatter(right[:, 0], right[:, 1], color="#DC2626", s=28, label="right")
    ax.set_title("3 groups in one layer", fontweight="bold")
    ax.set_xlabel("hidden feature 1")
    ax.set_ylabel("hidden feature 2")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True, loc="upper right")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(0.4, 4.25, "How the 3-class probe is read", fontsize=12.5, weight="bold")
    ax.text(0.4, 3.55, "The classifier tries to keep:", fontsize=10.5, color="#374151")
    ax.text(0.8, 2.95, "left away from center/right", fontsize=10.5, color="#2563EB", weight="bold")
    ax.text(0.8, 2.35, "center away from left/right", fontsize=10.5, color="#6B7280", weight="bold")
    ax.text(0.8, 1.75, "right away from left/center", fontsize=10.5, color="#DC2626", weight="bold")
    ax.text(0.4, 0.85, "Macro AUROC = average separability across all three one-vs-rest views.", fontsize=10.2, color="#111827")
    ax.text(0.4, 0.35, "High macro AUROC means the layer keeps the three ideology labels more distinct.", fontsize=10.2, color="#059669", weight="bold")

    st.pyplot(fig, use_container_width=True)


def plot_center_projection_only():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 3.4), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    left = (1.1, 1.2)
    right = (8.9, 1.2)
    center = (5.0, 2.9)
    proj = (5.0, 1.2)

    ax.annotate("", xy=right, xytext=left, arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": "#111827"})
    ax.scatter(*left, s=120, color="#2563EB")
    ax.scatter(*right, s=120, color="#DC2626")
    ax.scatter(*center, s=120, color="#6B7280")
    ax.scatter(*proj, s=90, color="#059669")
    ax.plot([center[0], proj[0]], [center[1], proj[1]], linestyle="--", color="#6B7280", lw=1.8)

    ax.text(left[0] - 0.15, left[1] - 0.45, "Left", color="#2563EB", fontsize=11)
    ax.text(right[0] - 0.2, right[1] - 0.45, "Right", color="#DC2626", fontsize=11)
    ax.text(center[0] - 0.35, center[1] + 0.2, "Center", color="#374151", fontsize=11)
    ax.text(proj[0] - 0.45, proj[1] + 0.22, "projection", color="#059669", fontsize=11)
    ax.text(0.55, 3.45, "Center projection on the Left->Right axis", fontsize=13, weight="bold")
    ax.text(0.6, 0.35, "This asks: if we collapse the center point down to the left-right axis, does it land closer to Left, closer to Right, or near the middle?", fontsize=10.2, color="#374151")

    st.pyplot(fig, use_container_width=True)


def plot_bias_direction_only():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.0, 4.6), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    left = (2.0, 2.0)
    center = (5.0, 4.0)
    right = (8.0, 2.0)

    ax.scatter(*left, s=130, color="#2563EB")
    ax.scatter(*center, s=130, color="#6B7280")
    ax.scatter(*right, s=130, color="#DC2626")
    ax.plot([left[0], center[0]], [left[1], center[1]], color="#2563EB", lw=2.2)
    ax.plot([center[0], right[0]], [center[1], right[1]], color="#DC2626", lw=2.2)

    ax.text(left[0] - 0.15, left[1] - 0.42, "Left", color="#2563EB", fontsize=11)
    ax.text(center[0] - 0.35, center[1] + 0.22, "Center", color="#374151", fontsize=11)
    ax.text(right[0] - 0.2, right[1] - 0.42, "Right", color="#DC2626", fontsize=11)

    ax.text(0.55, 4.72, "Bias direction = asymmetry of the center point", fontsize=13.5, weight="bold")
    ax.text(0.55, 4.35, r"$\mathrm{bias\ direction} = \angle(L,C) - \angle(R,C)$", fontsize=15, color="#111827")
    ax.text(0.55, 0.95, "If value ≈ 0", fontsize=10.8, weight="bold", color="#111827")
    ax.text(1.95, 0.95, "center is geometrically balanced between left and right", fontsize=10.6, color="#374151")
    ax.text(0.55, 0.52, "If value > 0", fontsize=10.8, weight="bold", color="#059669")
    ax.text(1.95, 0.52, "center is farther from left, so relatively closer to right", fontsize=10.6, color="#374151")
    ax.text(0.55, 0.10, "If value < 0", fontsize=10.8, weight="bold", color="#DC2626")
    ax.text(1.95, 0.10, "center is farther from right, so relatively closer to left", fontsize=10.6, color="#374151")

    st.pyplot(fig, use_container_width=True)


st.set_page_config(
    page_title="Political Bias RepEng – Meeting with Gianluca",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Political Bias Representation Engineering")
st.subheader("Kick-off Meeting with Gianluca – 31 Mar 2026")
st.markdown("---")

page = st.sidebar.selectbox(
    "选择页面",
    ["1. WHY → Motivation", "2. HOW → Pipeline", "3. WHAT → Results"],
    index=0,
)

if page == "1. WHY → Motivation":
    st.markdown("## 1. WHY → Motivation")
    st.markdown(
        """
The project starts from a simple concern:

**Political biases is not a single and stable entity in LLMs, but rather a collection of outputs derived from various measurement strategies.**

So the project asks a representation-level question:

**How is politically relevant information encoded inside the model when it processes political content?**

This matters for at least three reasons:
- political skew affects public-facing model behavior on controversial issues
- internal analysis is more informative than looking only at generated text
- if political signal has a stable internal structure, later intervention may become possible
"""
    )
elif page == "2. HOW → Pipeline":
    st.markdown("## 2. HOW → Pipeline")
    st.markdown(
        """
The project adapts ideas from representation engineering and safety-layer analysis to the political-bias setting.

The high-level idea is:
- first identify whether political content forms a distinct internal signal
- then study how Left / Right / Neutral are arranged within that signal
- then refine the analysis at the topic level
- finally attempt steering interventions
"""
    )
    st.markdown("")
    st.markdown("### Model Suite")
    st.markdown(
        """
| Family | Pre-trained | Instruction / Chat |
|------|-------------|--------------------|
| Qwen | Qwen2.5-7B | Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct |
| Mistral | Mistral-7B-v0.1 | Mistral-7B-Instruct-v0.2 |
| Llama | — | Llama2-Chinese-7B-Chat |
| ChatGLM | — | ChatGLM3-6B |
"""
    )
    st.markdown("")
    st.markdown("### Prompt Template")
    st.markdown(
        """
Before being fed into the model, each statement is wrapped in a unified instruction-style prompt:

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: {statement}
### Response:
```

This prompt is best described as an **Alpaca-style instruction template**, rather than a prompt originally introduced by Safety Layers. More precisely, it follows the Stanford Alpaca format and is also consistent with the template family adopted in the Safety Layers codebase.

References:
- Stanford Alpaca (instruction template used for fine-tuning and inference): https://github.com/tatsu-lab/stanford_alpaca#data-release
- Safety Layers (ICLR 2025) codebase includes Alpaca-style prompt templates (`alpaca.json` / `alpaca_legacy.json`): https://github.com/listen0425/Safety-Layers/tree/master/Code/templates
- Safety Layers paper (arXiv): https://arxiv.org/abs/2408.17003

We use this template for three reasons:
- it matches the instruction-following format used by many chat-tuned LLMs
- it removes formatting variation as a confounding factor
- the hidden state right before generation, after `### Response:`, is especially informative for representation analysis
"""
    )
    st.markdown("")
    st.markdown("### Step 1: Cosine Similarity -> Angular Gap")
    slide = st.radio(
        "Slides",
        [
            "1) Cosine as angle",
            "2) Why three pairings",
            "3) Angular gap + how to read",
        ],
        horizontal=True,
    )
    show_notes = st.sidebar.toggle("Presenter notes", value=True)

    if slide == "1) Cosine as angle":
        st.subheader("Slide 1 — Cosine similarity as an angle")
        plot_cosine_angle()
        st.markdown(
            "- Hidden state at a layer is a vector.\n"
            "- Cosine similarity measures direction similarity (scale-invariant).\n"
            "- We convert cosine to an angle for intuition: small θ ⇒ similar, large θ ⇒ dissimilar."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "一句话：把每层的 hidden state 当成向量，cosine 就是夹角的 cos。\n\n"
                    "你可以说：我们不关心向量长度，只关心方向是否一致，所以用 cosine。"
                )
    elif slide == "2) Why three pairings":
        st.subheader("Slide 2 — Why we need three pairings (P–P, NP–NP, P–NP)")
        plot_pairing_contrast()
        st.markdown(
            "- Cross-class similarity alone is not enough (layers differ in overall variance).\n"
            "- We subtract a within-class baseline: average of P–P and NP–NP.\n"
            "- If a layer encodes politicalness: θ(P–NP) should be larger than θ(P–P) and θ(NP–NP)."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "一句话：必须有对照，否则你不知道 P–NP 大是因为“政治性”，还是因为这一层本来就把所有句子都拉开了。\n\n"
                    "所以用 (P–P 与 NP–NP) 作为 baseline。"
                )
    elif slide == "3) Angular gap + how to read":
        st.subheader("Slide 3 — Angular gap definition + how to read the curve")
        st.markdown(
            "- We compare cross-class separation to within-class baselines.\n"
            "- If the curve rises, the layer is separating political from non-political more strongly.\n"
            "- Peaks indicate layers where political signal is most geometrically visible."
        )
        plot_gap_curve_example()
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "一句话：gap 是“跨类差异”减去“同类差异基线”。\n\n"
                    "读图方式：看 gap 曲线在哪些 layer 明显抬升/峰值，这些层更可能承载政治性信号。"
                )
    st.markdown("")
    st.markdown("### Step 1b: Logistic Regression Probe")
    probe_slide = st.radio(
        "Probe slides",
        [
            "1) What goes into the probe",
            "2) What 5-fold stratified CV does",
            "3) How to read accuracy and AUROC",
        ],
        horizontal=True,
    )

    if probe_slide == "1) What goes into the probe":
        st.subheader("Probe Slide 1 — What exactly is being classified?")
        plot_probe_input_representation()
        st.markdown(
            "- The LLM is frozen and only used as a feature extractor.\n"
            "- For one layer at a time, each text becomes one hidden-state vector.\n"
            "- These vectors, plus their labels, form the dataset for one logistic-regression probe."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "一句话：这里训练的不是 LLM，而是一个很小的分类器。\n\n"
                    "每一层都单独拿出来做一次，所以最后得到的是 layer-wise probe。"
                )
    elif probe_slide == "2) What 5-fold stratified CV does":
        st.subheader("Probe Slide 2 — Why do we split into 5 folds?")
        plot_probe_workflow()
        st.markdown(
            "- We split the data into 5 parts.\n"
            "- Each round uses 4 folds for training and 1 fold for testing.\n"
            "- 'Stratified' means each fold keeps the class balance.\n"
            "- We average the 5 test results to get a stable score for that layer."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "一句话：不是只做一次 train/test split，而是轮流测试 5 次。\n\n"
                    "这样结果不会依赖某一次刚好运气好的划分。"
                )
    else:
        st.subheader("Probe Slide 3 — AUROC (what it means in probing)")
        plot_probe_metrics_concept()
        st.markdown(
            "- The probe outputs a real-valued score for the label `political` at a given layer.\n"
            "- AUROC is threshold-free: it only cares about ranking between political vs non-political texts.\n"
            "- Practical reading: how often does a political text get a higher score than a non-political text?"
        )
        st.latex(r"\mathrm{AUROC} = P(X_P > X_{NP}) + \tfrac{1}{2} P(X_P = X_{NP})")
        st.latex(r"\mathrm{AUROC} = \frac{U}{n_P n_{NP}}\;\;\;(\text{Mann–Whitney }U\text{ / Wilcoxon rank-sum equivalence})")
        with st.expander("Accuracy (optional)", expanded=False):
            plot_accuracy_concept()
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "先说一句最关键的：AUROC 是 ranking 指标，不需要选阈值。\n\n"
                    "把 score 解释清楚：它是 logistic regression probe 对 `political` 标签输出的连续分数/置信度。\n\n"
                    "讲解顺序：\n"
                    "1) 看左图两类 score 的重叠：重叠越小，越容易分开。\n"
                    "2) AUROC 的一句话定义：随机抽一条 P 和一条 NP，P 的分数更高的概率。\n"
                    "3) 技术补充（给教授）：AUROC 等价于 Mann–Whitney U 的归一化，也等价于 ROC 曲线下面积。"
                )

    st.info(
        """
**Step 1 implementation snapshot**
- Public main run: `100 political + 100 non-political`
- Hidden-state readout: `last-token hidden state` at every layer
- Cosine analysis: `random pair mode`, `500 rounds`
- Probe analysis: one logistic-regression probe per layer, `5-fold stratified CV`
- Significance check: focused permutation test with `200` label shuffles
"""
    )

    st.markdown("")
    st.markdown("### Step 2: Ideology Analysis")
    step2_slide = st.radio(
        "Step 2 slides",
        [
            "1) What goes into the Step 2 probe",
            "2) What 3-class probing means",
            "3) How bias direction is computed",
            "4) How center projection is read",
        ],
        horizontal=True,
    )

    if step2_slide == "1) What goes into the Step 2 probe":
        st.subheader("Step 2 Slide 1 — What exactly is being classified?")
        plot_step2_input_representation()
        st.markdown(
            "- Step 2 uses three labels: left, center, and right.\n"
            "- The LLM is still frozen; we only read one layer at a time.\n"
            "- At each layer, the vectors plus the three ideology labels form one 3-class probe dataset."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "一句话：Step 2 和 Step 1 的形式是一样的，只是标签从二分类变成了三分类。\n\n"
                    "我们还是不训练 LLM，只训练一个很小的 probe。"
                )
    elif step2_slide == "2) What 3-class probing means":
        st.subheader("Step 2 Slide 2 — How do we read a 3-class probe?")
        plot_step2_probe_concept()
        st.markdown(
            "- We ask whether one layer keeps left, center, and right distinguishable.\n"
            "- The main score is macro AUROC: average separability across the three one-vs-rest views.\n"
            "- Higher macro AUROC means the ideology labels stay more distinct in that layer."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "可以先不用讲 one-vs-rest 的技术细节。\n\n"
                    "直接说：三分类 probe 在问这一层能不能把 left / center / right 三类区分开。"
                )
    elif step2_slide == "3) How bias direction is computed":
        st.subheader("Step 2 Slide 3 — How is bias direction computed?")
        plot_bias_direction_only()
        st.markdown(
            "- First compute the left, center, and right centroids at one layer.\n"
            "- Then compare the two angles: `angle(L, C)` and `angle(R, C)`.\n"
            "- Bias direction is their difference: `angle(L, C) - angle(R, C)`.\n"
            "- Near zero means center is more symmetric; positive or negative values mean center is closer to one side."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "这一页的重点不是讲复杂公式，而是讲它在问什么。\n\n"
                    "它在问：center 在几何上是不是对称地位于左右之间，还是更靠向某一边。"
                )
    else:
        st.subheader("Step 2 Slide 4 — How should we read center projection?")
        plot_center_projection_only()
        st.markdown(
            "- Draw the left-right axis first.\n"
            "- Then drop the center point down onto that axis.\n"
            "- This tells us whether center sits closer to left, closer to right, or near the middle in that layer."
        )
        if show_notes:
            with st.expander("讲解要点", expanded=True):
                st.markdown(
                    "一句话：projection 是把 center 在 left-right 方向上压成一个一维位置。\n\n"
                    "如果它总偏左或偏右，就说明这个 layer 里的 center 不是稳定中点。"
                )

    st.info(
        """
**Step 2 implementation snapshot**
- Public runs use `100 left + 100 center + 100 right`
- Text mode: headline / title only in the current public experiments
- Hidden-state readout: `last-token hidden state` at every layer
- Geometry: compute `left`, `center`, `right` centroids at each layer
- Bias direction: `angle(L, C) - angle(R, C)`
- Probe analysis: one 3-class logistic-regression probe per layer, `5-fold stratified CV`
- Main probe metric: `macro AUROC`
"""
    )
else:
    st.markdown("## 3. WHAT → Results")
    st.markdown("### Step 1 only: each dataset paired with examples, results, and interpretation")

    tabs = st.tabs([
        "Custom matched corpus",
        "political_or_not",
        "HuffPost (original)",
        "HuffPost strict + headline-only",
        "HuffPost lifestyle",
        "BBC",
        "Step 1 summary",
    ])

    with tabs[0]:
        st.subheader("Dataset 0 — Custom matched corpus (original starting point)")
        st.markdown("**Examples**")
        render_custom_matched_examples()
        st.markdown("**Results**")
        img = OLD_STEP1_COMBINED_COSINE
        if img.exists():
            st.image(
                str(img),
                caption="Original multi-model cosine-similarity comparison on the custom matched corpus",
                use_container_width=True,
            )
        st.markdown("**Interpretation**")
        st.info(
            "This was the earliest controlled dataset: left and right statements were topic-matched, and neutral statements were written for the same topics. Here I show the original comparison figure across models, which was the earliest cosine-similarity result used to see whether political and non-political representations separate at all."
        )
        st.warning(
            "This dataset is useful as the conceptual starting point, but not as the final main result. It is small and highly controlled, so later public-dataset experiments were needed for a more realistic Step 1 validation."
        )

    with tabs[1]:
        st.subheader("Dataset 1 — political_or_not")
        st.markdown("**Examples**")
        render_preview_examples(STEP1_POL_OR_NOT / "dataset" / "sample_preview.txt")
        st.markdown("**Results**")
        img = STEP1_POL_OR_NOT / "comparison" / "five_models_angular.png"
        if img.exists():
            st.image(str(img), caption="political_or_not: angular-gap profiles", use_container_width=True)
        st.markdown(peak_gap_markdown(STEP1_POL_OR_NOT / "comparison" / "five_models_summary.csv"))
        st.markdown("**Interpretation**")
        st.info(
            "This was the first public political/non-political dataset. It produced some signal, but the results were inconsistent, especially for Mistral. Random pairing versus all-pairs did not materially change the picture."
        )
        st.warning(
            "Main lesson: the political/non-political boundary in this dataset is too fuzzy for a clean cosine-based separation story."
        )

    with tabs[2]:
        st.subheader("Dataset 2 — HuffPost (initial version)")
        st.markdown("**Examples**")
        render_preview_examples(STEP1_HUFFPOST_ORIG / "dataset" / "sample_preview.txt")
        st.markdown("**Results**")
        img = STEP1_HUFFPOST_ORIG / "comparison" / "five_models_angular.png"
        if img.exists():
            st.image(str(img), caption="HuffPost initial: angular-gap profiles", use_container_width=True)
        st.markdown(peak_gap_markdown(STEP1_HUFFPOST_ORIG / "comparison" / "five_models_summary.csv"))
        st.markdown("**Interpretation**")
        st.info(
            "This version used `POLITICS` versus `SPORTS + ENTERTAINMENT + TRAVEL + WELLNESS`. The signal was weak because the negative class still contained semi-political content, especially from entertainment."
        )

    with tabs[3]:
        st.subheader("Dataset 3 — HuffPost strict + headline-only")
        st.markdown("**Examples**")
        render_preview_examples(STEP1_STRICT / "dataset" / "sample_preview.txt")
        st.markdown("**Results**")
        img = STEP1_STRICT / "comparison" / "five_models_angular.png"
        if img.exists():
            st.image(str(img), caption="HuffPost strict headline-only: angular-gap profiles", use_container_width=True)
        st.markdown(peak_gap_markdown(STEP1_STRICT / "comparison" / "five_models_summary.csv"))
        st.markdown("**Interpretation**")
        st.info(
            "Removing `ENTERTAINMENT` and keeping only headlines made the signal much cleaner. This confirmed that negative-class definition and text length strongly affect cosine-based politicality analysis."
        )

    with tabs[4]:
        st.subheader("Dataset 4 — HuffPost lifestyle (final Step 1 main dataset)")
        st.markdown("**Examples**")
        render_preview_examples(STEP1_LIFESTYLE / "dataset" / "sample_preview.txt")

        st.markdown("**Cosine / angular result**")
        col1, col2 = st.columns(2)
        with col1:
            img = STEP1_LIFESTYLE / "comparison" / "five_models_angular.png"
            if img.exists():
                st.image(str(img), caption="HuffPost lifestyle: angular-gap profiles", use_container_width=True)
        with col2:
            img = STEP1_LIFESTYLE / "comparison" / "five_models_cosine.png"
            if img.exists():
                st.image(str(img), caption="HuffPost lifestyle: cosine profiles", use_container_width=True)
        st.markdown(peak_gap_markdown(STEP1_LIFESTYLE / "comparison" / "five_models_summary.csv"))

        st.markdown("**Logistic-regression result**")
        probe_img = STEP1_LIFESTYLE / "comparison" / "five_models_probe.png"
        if probe_img.exists():
            st.image(str(probe_img), caption="HuffPost lifestyle: layer-wise logistic regression probe", use_container_width=True)
        st.markdown(
            """
**What this figure is saying**

Each panel is one model. The x-axis is the layer index, and the y-axis is probe performance
(accuracy / AUROC). The main pattern is that the curves stay **high across many layers**.
So the most important message is not "there is one special peak layer", but rather:
**political vs non-political information can be read out reliably from a broad part of the network.**
"""
        )
        st.info(
            "Practical reading: this figure is evidence that the signal clearly exists and is strong. "
            "It is showing broad decodability, not a sharp one-layer localization result."
        )
        st.warning(
            "What not to over-claim from this figure alone: it does not by itself prove a precise set of "
            "`political layers`, and it does not by itself prove pure politicality rather than dataset-specific signal."
        )
        st.markdown(
            """
| Model | Peak accuracy | Peak AUROC |
|---|---:|---:|
| Qwen/Qwen2.5-7B | **97.5%** | 0.993 |
| mistralai/Mistral-7B-v0.1 | **97.5%** | 0.999 |
| Qwen/Qwen2.5-7B-Instruct | **97.0%** | 0.993 |
| mistralai/Mistral-7B-Instruct-v0.2 | **94.0%** | 0.985 |
| Qwen/Qwen2.5-1.5B-Instruct | **94.0%** | 0.984 |
"""
        )

        st.markdown("**Interpretation**")
        st.success(
            "This became the main Step 1 dataset. The cosine result says the geometric gap is visible but modest. The logistic-regression result adds the crucial point: political vs non-political information is not weak at all — it is strongly and stably linearly decodable across models."
        )
        st.warning(
            "Important caveat: because this is still a proxy politicality setting, the probe result should be read as strong decodability under this dataset definition, not as a final proof of pure politicality."
        )

    with tabs[5]:
        st.subheader("Dataset 5 — BBC")
        st.markdown("**Examples**")
        render_preview_examples(STEP1_BBC / "dataset" / "sample_preview.txt")
        st.markdown("**Results**")
        img = STEP1_BBC / "comparison" / "five_models_angular.png"
        if img.exists():
            st.image(str(img), caption="BBC: angular-gap profiles", use_container_width=True)
        st.markdown(peak_gap_markdown(STEP1_BBC / "comparison" / "five_models_summary.csv"))
        st.markdown("**Interpretation**")
        st.info(
            "BBC works as a single-source supplementary check. It produces a signal, but it is clearly weaker than HuffPost lifestyle."
        )

    with tabs[6]:
        st.subheader("Step 1 summary")
        st.markdown(
            """
### Summary: Dataset Search
- The result is highly sensitive to how the non-political class is defined.
- Cleaner and more obviously non-political negatives produce stronger cosine separation.
- HuffPost lifestyle is the strongest public Step 1 proxy among the datasets tested.

### Summary: Methods Comparison
- **Cosine similarity / angular gap** gives a useful geometric signature.
- But the gap is still only a few degrees, so cosine alone is not enough.
- **Logistic regression** on the same HuffPost lifestyle data gives much stronger evidence.

### Question for Discussion
- **How should we frame Step 1:** as broad linear decodability of political information, or as localization of political layers?
- **Is HuffPost lifestyle an acceptable main proxy dataset for Step 1, or do we need a more controlled dataset before making stronger claims?**

These two questions matter because the current results are already strong enough to support a real Step 1 finding, but the exact interpretation and the acceptable strength of the claim still need to be agreed on.
"""
        )

    st.markdown("---")
    st.markdown("## Step 2 Results")
    st.markdown("### Public ideology datasets: examples, results, and framing")

    step2_tabs = st.tabs([
        "Article-Bias-Prediction",
        "AllSides",
        "Step 2 summary",
    ])

    with step2_tabs[0]:
        st.subheader("Dataset 6 — Article-Bias-Prediction")
        st.markdown("**Examples**")
        render_threeway_preview_examples(STEP2_DIR / "outputs" / "sample_preview.txt")

        st.markdown("**Results**")
        col1, col2 = st.columns(2)
        with col1:
            img = STEP2_COMPARISON / "five_models_public_step2_probe.png"
            if img.exists():
                st.image(str(img), caption="Article-Bias: 3-class probe — can each layer decode left / center / right?", use_container_width=True)
        with col2:
            img = STEP2_COMPARISON / "five_models_public_step2_bias_direction.png"
            if img.exists():
                st.image(str(img), caption="Article-Bias: bias-direction geometry — is center closer to one side?", use_container_width=True)
        img = STEP2_COMPARISON / "five_models_public_step2_center_projection.png"
        if img.exists():
            st.image(str(img), caption="Article-Bias: center projection — where does center lie on the left-right axis?", use_container_width=True)
        render_step2_figure_guide()
        st.markdown(step2_probe_markdown(STEP2_COMPARISON / "five_models_public_step2_summary.csv"))

        st.markdown("**Interpretation**")
        st.info(
            "Article-Bias produces a strong ideology-decoding result under the current pipeline. "
            "The 3-class probe reaches high AUROC across all five models."
        )
        st.warning(
            "However, this result is not cleanly interpretable as ideology alone. The left / center / right sources are highly non-overlapping, "
            "so source and writing-style confounds are a serious concern."
        )

    with step2_tabs[1]:
        st.subheader("Dataset 7 — AllSides (topic-matched)")
        st.markdown("**Examples**")
        render_allsides_preview(STEP2_DIR / "outputs_allsides" / "sample_preview.txt")

        st.markdown("**Results**")
        col1, col2 = st.columns(2)
        with col1:
            img = STEP2_ALLSIDES / "five_models_public_step2_probe.png"
            if img.exists():
                st.image(str(img), caption="AllSides: 3-class probe — can each layer decode left / center / right?", use_container_width=True)
        with col2:
            img = STEP2_ALLSIDES / "five_models_public_step2_bias_direction.png"
            if img.exists():
                st.image(str(img), caption="AllSides: bias-direction geometry — is center closer to one side?", use_container_width=True)
        img = STEP2_ALLSIDES / "five_models_public_step2_center_projection.png"
        if img.exists():
            st.image(str(img), caption="AllSides: center projection — where does center lie on the left-right axis?", use_container_width=True)
        render_step2_figure_guide()
        st.markdown(step2_probe_markdown(STEP2_ALLSIDES / "five_models_public_step2_summary.csv"))
        st.markdown(step2_compare_markdown(STEP2_ALLSIDES / "allsides_vs_article_bias_summary.csv"))

        st.markdown("**Interpretation**")
        st.info(
            "AllSides is the stricter dataset because it matches left / center / right at the event level. "
            "So it is a better stress test for whether the ideology result survives topic control."
        )
        st.warning(
            "Under this stricter setting, the signal becomes much weaker. This suggests that the stronger Article-Bias result is at least partly driven by source / topic / style confounds."
        )

    with step2_tabs[2]:
        st.subheader("Step 2 summary")
        st.markdown(
            """
### Summary: Dataset Issues

**Article-Bias-Prediction**
- Strength: it gives a strong left / center / right decoding result under the current pipeline.
- Main issue: the three labels come from largely non-overlapping sources.
- So the strong result may reflect source and writing-style fingerprints, not ideology alone.

**AllSides**
- Strength: it is much stricter because it is event-matched across left / center / right.
- Main issue: under this stricter setting, the signal becomes much weaker.
- So it is better controlled, but it does not currently give a strong ideology-decoding result with the present method.

### Overall Step 2 Takeaway
- The current Step 2 result is highly sensitive to dataset design.
- One dataset is stronger but heavily confounded; the other is cleaner but much weaker.
- So Step 2 currently works better as a diagnostic comparison than as a final localization claim about ideology layers.

### Question for Discussion
- **How should we frame Step 2:** as an exploratory ideology analysis, or mainly as a confound diagnosis?
- **Do we need a more controlled ideology dataset before making stronger claims about ideology-specific layers?**

These questions matter because Step 2 is already informative, but the main limitation is now clear: dataset choice changes the interpretation of the ideology signal.
"""
        )
