"""
Political Bias Representation Engineering – Results Summary

Companion Streamlit app for collaborator updates.
Keeps the original WHY / HOW / WHAT framing, but organizes WHAT
chronologically by dataset and method pivots.
"""

from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RESULTS_ROOT = BASE.parent / "political-bias-representation-engineering"

STEP1 = RESULTS_ROOT / "step1_politicality_v2"
STEP2 = RESULTS_ROOT / "step2_ideology_public_v1"

STEP1_LIFESTYLE = STEP1 / "outputs_huffpost_lifestyle_random"
STEP1_STRICT = STEP1 / "outputs_huffpost_random_strict_headline_only"
STEP1_BBC = STEP1 / "outputs_bbc_random"
STEP1_DATASET_CMP = STEP1 / "outputs_dataset_comparison"

STEP2_AB = STEP2 / "comparison"
STEP2_ALLSIDES = STEP2 / "comparison_allsides"
STEP2_ABLATION = STEP2 / "route_b_allsides_ablation"


# ---------------------------------------------------------------------------
# Small plotting helpers for HOW slides
# ---------------------------------------------------------------------------
def plot_cosine_angle() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    fig, ax = plt.subplots(figsize=(9.0, 3.3), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(0, 6.1)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    origin = (0.7, 0.6)
    v_pol = (4.2, 1.4)
    v_non = (3.1, 2.7)

    ax.annotate("", xy=(5.5, origin[1]), xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 2.4})
    ax.annotate("", xy=(origin[0], 3.1), xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 2.4})

    ax.annotate("", xy=v_pol, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": "#2563eb"})
    ax.annotate("", xy=v_non, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": "#f97316"})

    ax.text(v_pol[0] + 0.1, v_pol[1] - 0.15, "v(P)", color="#2563eb", fontsize=13, weight="bold")
    ax.text(v_non[0] + 0.1, v_non[1] + 0.05, "v(NP)", color="#f97316", fontsize=13, weight="bold")

    arc = Arc(origin, width=2.1, height=2.1, angle=0, theta1=10, theta2=44, lw=2.2)
    ax.add_patch(arc)
    ax.text(origin[0] + 1.2, origin[1] + 0.7, "theta", fontsize=13)

    box = dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC", edgecolor="#E5E7EB", linewidth=1.2)
    ax.text(
        3.95,
        3.0,
        "Cosine -> angle\nsame direction = small angle",
        fontsize=13,
        bbox=box,
        va="top",
    )
    ax.text(4.05, 2.05, "Small theta -> similar", fontsize=12)
    ax.text(4.05, 1.72, "Large theta -> dissimilar", fontsize=12)

    st.pyplot(fig, use_container_width=True)


def plot_pairing_contrast() -> None:
    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), constrained_layout=True)
    titles = [
        "Within-class: P-P (small theta)",
        "Within-class: NP-NP (small theta)",
        "Cross-class: P-NP (large theta)",
    ]
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

        rad1 = math.radians(deg1)
        rad2 = math.radians(deg2)
        v1 = (origin[0] + length * math.cos(rad1), origin[1] + length * math.sin(rad1))
        v2 = (origin[0] + length * math.cos(rad2), origin[1] + length * math.sin(rad2))

        ax.annotate("", xy=v1, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": c1})
        ax.annotate("", xy=v2, xytext=origin, arrowprops={"arrowstyle": "-|>", "lw": 3, "color": c2})

        theta1 = min(deg1, deg2)
        theta2 = max(deg1, deg2)
        radius = 1.35
        arc = Arc(origin, width=2 * radius, height=2 * radius, angle=0, theta1=theta1, theta2=theta2, lw=2.2)
        ax.add_patch(arc)

        mid = math.radians((theta1 + theta2) / 2.0)
        ax.text(
            origin[0] + (radius + 0.25) * math.cos(mid),
            origin[1] + (radius + 0.25) * math.sin(mid),
            "theta",
            fontsize=11,
            ha="center",
            va="center",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")

    st.pyplot(fig, use_container_width=True)


def plot_gap_curve_example() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    layers = np.arange(1, 33)
    peak = 12
    gap = 4 + 18 * np.exp(-((layers - peak) ** 2) / (2 * 18)) + 1.2 * np.exp(-((layers - 28) ** 2) / (2 * 10))
    fig, ax = plt.subplots(figsize=(9.5, 2.8), constrained_layout=True)
    ax.plot(layers, gap, color="#111827", lw=2.2)
    ax.axvline(peak, color="#ef4444", lw=1.5, linestyle="--")
    ax.text(peak + 0.4, float(gap.max()) - 1.0, "peak", color="#ef4444", fontsize=11)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Angular gap (deg)")
    ax.set_title("Illustration: where political vs non-political separates most", fontweight="bold")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def plot_logistic_regression_concept() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(7)
    pol = rng.normal(loc=(1.8, 2.2), scale=(0.22, 0.28), size=(24, 2))
    non = rng.normal(loc=(3.1, 1.2), scale=(0.28, 0.24), size=(24, 2))

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
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


def plot_permutation_concept() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(11)
    null_scores = rng.normal(0.56, 0.025, 200)
    observed = 0.95

    fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
    ax.hist(null_scores, bins=18, color="#CBD5E1", edgecolor="#94A3B8")
    ax.axvline(null_scores.mean(), color="#64748B", lw=1.6, linestyle="--", label="null mean")
    ax.axvline(np.quantile(null_scores, 0.95), color="#0F766E", lw=1.6, linestyle="--", label="null 95th pct")
    ax.axvline(observed, color="#DC2626", lw=2.4, label="observed probe")
    ax.text(observed + 0.005, ax.get_ylim()[1] * 0.78, "far above\nshuffled labels", color="#DC2626", fontsize=11)
    ax.set_title("Permutation test: compare the real probe score to shuffled-label runs", fontweight="bold")
    ax.set_xlabel("probe score")
    ax.set_ylabel("count")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.15)
    st.pyplot(fig, use_container_width=True)


def plot_three_class_probe_concept() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(17)
    left = rng.normal(loc=(1.2, 1.1), scale=(0.18, 0.18), size=(18, 2))
    center = rng.normal(loc=(2.2, 2.2), scale=(0.18, 0.18), size=(18, 2))
    right = rng.normal(loc=(3.2, 1.2), scale=(0.18, 0.18), size=(18, 2))

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    ax.scatter(left[:, 0], left[:, 1], color="#2563eb", s=34, label="left")
    ax.scatter(center[:, 0], center[:, 1], color="#6B7280", s=34, label="center")
    ax.scatter(right[:, 0], right[:, 1], color="#DC2626", s=34, label="right")

    ax.text(0.95, 0.72, "left", color="#2563eb", fontsize=11)
    ax.text(2.05, 2.55, "center", color="#374151", fontsize=11)
    ax.text(3.1, 0.77, "right", color="#DC2626", fontsize=11)

    ax.set_title("3-class probing: can a layer distinguish left, center, and right?", fontweight="bold")
    ax.set_xlabel("hidden feature 1")
    ax.set_ylabel("hidden feature 2")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig, use_container_width=True)


def plot_center_projection_concept() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    left = (1.4, 1.1)
    right = (8.6, 1.1)
    center = (5.1, 2.8)
    proj = (5.1, 1.1)

    ax.annotate("", xy=right, xytext=left, arrowprops={"arrowstyle": "-|>", "lw": 2.8, "color": "#111827"})
    ax.scatter(*left, s=120, color="#2563eb")
    ax.scatter(*right, s=120, color="#DC2626")
    ax.scatter(*center, s=120, color="#6B7280")
    ax.scatter(*proj, s=85, color="#059669")
    ax.plot([center[0], proj[0]], [center[1], proj[1]], linestyle="--", color="#6B7280", lw=1.8)

    ax.text(left[0] - 0.2, left[1] - 0.45, "left centroid", color="#2563eb", fontsize=11)
    ax.text(right[0] - 0.4, right[1] - 0.45, "right centroid", color="#DC2626", fontsize=11)
    ax.text(center[0] - 0.5, center[1] + 0.25, "center centroid", color="#374151", fontsize=11)
    ax.text(proj[0] - 0.5, proj[1] + 0.25, "projection", color="#059669", fontsize=11)
    ax.set_title("Center projection: where does the center representation fall on the left-right axis?", fontweight="bold")
    st.pyplot(fig, use_container_width=True)


def plot_candidate_layers_concept() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    layers = np.arange(1, 33)
    probe = 0.55 + 0.4 * np.exp(-((layers - 24) ** 2) / (2 * 18))
    gap = 0.8 + 2.7 * np.exp(-((layers - 26) ** 2) / (2 * 10))
    band_start, band_end = 22, 27

    fig, ax1 = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.axvspan(band_start, band_end, color="#FEF3C7", alpha=0.55, label="candidate band")
    ax1.plot(layers, probe, color="#2563eb", lw=2.4, label="probe AUROC")
    ax2.plot(layers, gap, color="#F97316", lw=2.4, label="angular gap")

    ax1.set_xlabel("layer")
    ax1.set_ylabel("probe", color="#2563eb")
    ax2.set_ylabel("gap", color="#F97316")
    ax1.set_title("Candidate layers = where probe and geometry both become strong", fontweight="bold")
    ax1.grid(True, alpha=0.18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    st.pyplot(fig, use_container_width=True)


def plot_probe_pipeline_flow() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.2, 4.6), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F8FAFC", ec="#CBD5E1", fontsize=11, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.12",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#111827"})

    # Row 1
    box(0.4, 3.9, 2.0, 1.0, "Political texts\n+\nNon-political texts", fc="#EFF6FF", ec="#93C5FD", weight="bold")
    box(3.0, 3.9, 2.1, 1.0, "Wrap with\nprompt template", fc="#F8FAFC")
    box(5.7, 3.9, 2.2, 1.0, "Forward pass\nthrough LLM", fc="#F8FAFC")
    box(8.5, 3.9, 2.3, 1.0, "Extract last-token\nhidden states", fc="#F8FAFC")
    box(11.4, 3.9, 2.0, 1.0, "For one layer l:\nbuild X_l and y_l", fc="#FEF3C7", ec="#F59E0B", weight="bold")

    arrow(2.4, 4.4, 3.0, 4.4)
    arrow(5.1, 4.4, 5.7, 4.4)
    arrow(7.9, 4.4, 8.5, 4.4)
    arrow(10.8, 4.4, 11.4, 4.4)

    # Row 2
    box(1.3, 1.2, 2.2, 1.0, "Split into 5\nstratified folds", fc="#ECFDF5", ec="#86EFAC", weight="bold")
    box(4.1, 1.2, 2.2, 1.0, "Train on 4 folds\nTest on 1 fold", fc="#F8FAFC")
    box(6.9, 1.2, 2.2, 1.0, "Standardize features\n(StandardScaler)", fc="#F8FAFC")
    box(9.7, 1.2, 2.2, 1.0, "Fit logistic\nregression", fc="#F8FAFC")
    box(12.2, 1.2, 1.4, 1.0, "Record\naccuracy\n+ AUROC", fc="#FCE7F3", ec="#F9A8D4", weight="bold")

    arrow(12.4, 3.9, 2.5, 2.2)
    arrow(3.5, 1.7, 4.1, 1.7)
    arrow(6.3, 1.7, 6.9, 1.7)
    arrow(9.1, 1.7, 9.7, 1.7)
    arrow(11.9, 1.7, 12.2, 1.7)

    # Final aggregation
    box(4.6, 0.1, 4.8, 0.7, "Repeat for every fold, average the scores, then repeat for every layer", fc="#EDE9FE", ec="#A78BFA", weight="bold")
    arrow(12.9, 1.2, 9.0, 0.8)

    ax.set_title("Layer-wise logistic-regression probe: what actually happens", fontsize=14, fontweight="bold")
    st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Missing figure: {path.name}")


def presenter_notes(text: str, enabled: bool) -> None:
    if enabled:
        with st.expander("Presenter notes", expanded=True):
            st.markdown(text)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Political Bias RepEng – Results Summary",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Political Bias Representation Engineering")
st.subheader("Results Summary for Collaborator Meeting")
st.markdown("---")


PAGES = [
    "1. WHY -> Motivation",
    "2. HOW -> Methods",
    "3. Dataset 1: Custom Matched Corpus",
    "4. Dataset 2: political_or_not",
    "5. Dataset 3: HuffPost (iterations)",
    "6. Dataset 4: BBC",
    "7. Dataset 5: ERCE (planned validation)",
    "8. Method Pivot: Linear Probe + Permutation",
    "9. Step 1 Integration: Candidate Political Layers",
    "10. Dataset 6: Article-Bias-Prediction",
    "11. Dataset 7: AllSides (topic-matched)",
    "12. AllSides Route B: Ablation",
    "13. Integrated Takeaways",
]

page = st.sidebar.selectbox("Navigate", PAGES, index=0)
show_notes = st.sidebar.toggle("Show presenter notes", value=True)


if page == PAGES[0]:
    st.markdown("## 1. WHY -> Motivation")
    st.markdown(
        """
The project starts from a simple concern:

> **Political bias is not a single stable object in LLMs, but a family of effects that depends on how we measure it.**

This motivates a representation-level question:

> **How is politically relevant information encoded inside the model when it processes political content?**

I split this into two sub-questions:

- **Politicality (Step 1):** does the model internally distinguish political from non-political text?
- **Ideology (Step 2):** within political text, does the model distinguish left / center / right?
"""
    )
    st.markdown(
        """
This matters because:

- output-only evaluation hides the internal structure of the model
- political bias claims can be method-dependent
- if the signal has stable internal structure, later intervention may be possible
"""
    )
    presenter_notes(
        "这一页只定问题，不讲结果。核心是把整个项目拆成两个层次：\n\n"
        "1. politicality：是不是政治\n"
        "2. ideology：政治上偏哪边\n\n"
        "后面所有实验都围绕这两个问题推进。",
        show_notes,
    )

elif page == PAGES[1]:
    st.markdown("## 2. HOW -> Methods")
    st.markdown("This page summarizes the shared analysis pipeline, independent of dataset choice.")

    st.markdown("### Model Suite")
    st.markdown(
        """
| Family | Pre-trained | Instruction-tuned |
|--------|-------------|-------------------|
| Qwen | Qwen2.5-7B | Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct |
| Mistral | Mistral-7B-v0.1 | Mistral-7B-Instruct-v0.2 |
"""
    )

    st.markdown("### Representation Extraction")
    st.markdown(
        """
- Unified Alpaca-style prompt template
- One forward pass per input
- Layer-wise hidden states extracted at every transformer block
- Default representation: **last-token hidden state**
- Default text mode for public datasets: **headline only**, unless noted otherwise
"""
    )

    st.markdown("### Prompt Template")
    st.code(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        "### Instruction: {statement}\n"
        "### Response:",
        language="text",
    )

    st.markdown("### Step 1 Methods (Politicality)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
**Cosine Similarity**
- pairwise similarity between hidden-state vectors
- compares direction, not magnitude
- starting point for geometric analysis
"""
        )
    with c2:
        st.markdown(
            """
**Angular Gap**
- convert cosine to angle with arccos
- compare `P-P`, `NP-NP`, `P-NP`
- asks whether political vs non-political separates geometrically
"""
        )
    with c3:
        st.markdown(
            """
**Linear Probe**
- per-layer logistic regression
- 5-fold stratified CV
- metrics: accuracy, balanced accuracy, AUROC
- used as the main sensitivity check
"""
        )

    st.markdown("### Visual explanation")
    slide = st.radio(
        "HOW mini-slides",
        [
            "Cosine as angle",
            "Why three pairings",
            "How angular gap is read",
            "Probe pipeline",
            "How logistic regression works",
            "How permutation works",
            "How 3-class probing works",
            "How center projection works",
            "How candidate layers are defined",
        ],
        horizontal=True,
    )
    if slide == "Cosine as angle":
        plot_cosine_angle()
        st.markdown(
            """
- each hidden state is treated as a vector
- cosine checks whether two vectors point in a similar direction
- angle is just the intuitive version of the same idea
"""
        )
    elif slide == "Why three pairings":
        plot_pairing_contrast()
        st.markdown(
            """
- `P-P` asks whether political texts cluster together
- `NP-NP` asks whether non-political texts cluster together
- `P-NP` asks whether the two groups separate from each other
"""
        )
    elif slide == "How angular gap is read":
        plot_gap_curve_example()
        st.markdown(
            """
- the curve rises when cross-class separation becomes larger than within-class variation
- peaks indicate where political vs non-political becomes most geometrically distinct
"""
        )
    elif slide == "Probe pipeline":
        plot_probe_pipeline_flow()
        st.markdown(
            """
- the LLM is used only as a **feature extractor**
- for each layer, I build one dataset `X_l, y_l`
- I then run 5-fold stratified cross-validation on that layer alone
- the final output is a layer-wise profile of accuracy and AUROC
"""
        )
    elif slide == "How logistic regression works":
        plot_logistic_regression_concept()
        st.markdown(
            """
- for one layer at a time, I ask whether a **simple linear boundary** can separate the classes
- if a layer gives high accuracy or AUROC, that means the information is linearly decodable there
- this is why logistic regression became the main sensitivity check
"""
        )
    elif slide == "How permutation works":
        plot_permutation_concept()
        st.markdown(
            """
- I shuffle the labels and rerun the same probe many times
- this builds a null distribution
- if the real result sits far to the right, the signal is unlikely to be random
"""
        )
    elif slide == "How 3-class probing works":
        plot_three_class_probe_concept()
        st.markdown(
            """
- Step 2 repeats the same layer-wise logic, but now for three classes: left, center, and right
- the question is no longer “is this political?”, but “which ideological side does this representation resemble?”
"""
        )
    elif slide == "How center projection works":
        plot_center_projection_concept()
        st.markdown(
            """
- left and right define an ideological axis
- I then ask where the center representation falls on that axis
- this gives a geometric view of whether center is actually centered or pulled to one side
"""
        )
    else:
        plot_candidate_layers_concept()
        st.markdown(
            """
- geometry alone is not enough
- probing alone is also not enough
- so I take the layers where both signals become strong and treat them as a candidate band
"""
        )

    st.markdown("### Statistical Validation")
    st.markdown(
        """
- **Permutation test** on the probe output
- shuffle labels 200 times
- compare observed performance against the null distribution
- answers: *is the signal non-random?*
"""
    )

    st.markdown("### Step 2 Methods (Ideology)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**3-class probing**
- left / center / right
- per-layer logistic regression
- macro AUROC as the main metric
"""
        )
    with c2:
        st.markdown(
            """
**Centroid geometry**
- left-center / right-center / left-right distances
- bias direction
- center projection on the left-to-right axis
"""
        )

    st.markdown("### Candidate Political Layers")
    st.markdown(
        """
- top-5 angular layers
- top-5 probe AUROC layers
- overlap plus +/-1 neighboring layers

This produces a **candidate band** per model, not a hard boundary.
"""
    )

    st.caption(
        "Early exploratory SVM / shallow MLP baselines are discussed later as part of the experimental history, "
        "but they are not retained in the final core pipeline."
    )
    presenter_notes(
        "HOW 这一页只讲通用方法，不讲具体数据集历史。\n\n"
        "重点是：\n"
        "1. cosine / angular gap 是几何读出\n"
        "2. logistic regression 是更灵敏的线性解码\n"
        "3. permutation 是统计显著性\n"
        "4. Step 2 用 3-class probe 和 centroid geometry\n\n"
        "现在这一页已经尽量改成图解版，可以少讲公式，多讲直觉。",
        show_notes,
    )

elif page == PAGES[2]:
    st.markdown("## Dataset 1: Custom Matched Corpus")
    st.markdown("### Why this dataset")
    st.markdown(
        """
This was the starting point: a small, manually controlled dataset with matched topics.
Its purpose was to test whether political signal could appear under ideal conditions.
"""
    )
    st.markdown("### Method used")
    st.markdown("- cosine / angular gap\n- early SVM and shallow MLP probing")
    st.markdown("### Result")
    st.markdown(
        """
- cosine gaps looked strong
- probing performance became near-perfect from very early layers
"""
    )
    st.warning(
        "Interpretation: the early probe results were likely inflated by overfitting in a low-sample, high-dimensional setting. "
        "This dataset suggested that some signal existed, but it could not support a reliable final conclusion."
    )
    presenter_notes(
        "这是起点。它的价值不是形成最终结果，而是告诉我：模型内部可能真的有 political signal。\n\n"
        "但旧版 probing 太不可信，所以必须转向更大的公开数据。",
        show_notes,
    )

elif page == PAGES[3]:
    st.markdown("## Dataset 2: political_or_not")
    st.markdown("### Why this dataset")
    st.markdown(
        "I moved to a public political-vs-non-political dataset to see whether the Step 1 signal survives on more realistic data."
    )
    st.markdown("### Method used")
    st.markdown("- cosine / angular gap\n- compared random pairing vs all-pairs")
    st.markdown("### Result")
    st.markdown(
        """
- signal remained weak
- random pairing and all-pairs gave very similar conclusions
"""
    )
    st.warning(
        "Interpretation: the main issue was not Monte Carlo pairing noise. The larger issue was that the political/non-political boundary in this dataset was too fuzzy for clean geometric separation."
    )
    presenter_notes(
        "这个数据集让我学到：如果数据定义本身不干净，pairing 再怎么改也救不回来。",
        show_notes,
    )

elif page == PAGES[4]:
    st.markdown("## Dataset 3: HuffPost (iterations)")
    st.markdown(
        "I stayed on HuffPost the longest, because it let me change the negative class definition while keeping the data source fixed."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Initial", "Strict", "Headline-only", "Lifestyle (main)"]
    )

    with tab1:
        st.markdown("### Initial HuffPost")
        st.markdown("- positive: `POLITICS`\n- negative: `SPORTS + ENTERTAINMENT + TRAVEL + WELLNESS`")
        st.info("Result: the signal was still weak and hard to interpret.")
        st.caption("Takeaway: the negative class likely contained semi-political content.")

    with tab2:
        st.markdown("### HuffPost Strict")
        st.markdown("Change: removed `ENTERTAINMENT` from the negative class.")
        st.info("Result: signal became noticeably stronger.")
        st.caption("Takeaway: negative-class purity matters directly.")

    with tab3:
        st.markdown("### Headline-only")
        st.markdown("Change: switched from `headline + short_description` to `headline_only`.")
        st.info("Result: 4 out of 5 models improved.")
        st.caption("Takeaway: shorter text works better for the current cosine-based public pipeline.")

    with tab4:
        st.markdown("### HuffPost Lifestyle (current Step 1 main result)")
        st.markdown(
            "- positive: `POLITICS`\n"
            "- negative: `FOOD & DRINK + STYLE & BEAUTY + TRAVEL + WELLNESS`\n"
            "- text mode: `headline_only`"
        )
        show_image(
            STEP1_LIFESTYLE / "comparison" / "five_models_angular.png",
            "Five models: angular gap across layers on HuffPost Lifestyle",
        )
        st.markdown(
            """
| Model | Peak angular gap | Peak layer |
|-------|------------------|------------|
| Qwen2.5-7B-Instruct | **4.156 deg** | 27 |
| Mistral-7B-Instruct-v0.2 | 3.509 deg | 31 |
| Qwen2.5-7B | 3.024 deg | 27 |
| Mistral-7B-v0.1 | 2.626 deg | 28 |
| Qwen2.5-1.5B-Instruct | 1.294 deg | 21 |
"""
        )
        st.success(
            "This became the strongest public Step 1 setting under the current pipeline."
        )
        st.warning(
            "Caveat: this remains a political-vs-lifestyle proxy setting. Strong separation here may partly reflect topic distance, not only pure politicality."
        )

    st.markdown("### Cross-iteration comparison")
    st.markdown(
        """
| Variant | Mean peak angular gap |
|--------|------------------------|
| HuffPost Lifestyle | **2.922 deg** |
| HuffPost Strict (headline-only) | 1.072 deg |
| BBC | 0.973 deg |
"""
    )
    presenter_notes(
        "HuffPost 这部分是 Step 1 真正的主结果线。\n\n"
        "重点讲清楚：我不是随便选了一个数据集，而是在同一数据源里不断清洗 negative class，最后找到最强配置。",
        show_notes,
    )

elif page == PAGES[5]:
    st.markdown("## Dataset 4: BBC")
    st.markdown("### Why this dataset")
    st.markdown("BBC served as a single-source supplementary validation for Step 1.")
    st.markdown("### Method used")
    st.markdown("- same cosine / angular-gap pipeline\n- politics vs sport + tech + entertainment")
    st.markdown("### Result")
    st.markdown(
        """
- signal existed
- overall separation was weaker than on HuffPost Lifestyle
"""
    )
    st.info(
        "Interpretation: BBC is useful as a supplement, but not strong enough to replace HuffPost Lifestyle as the current public Step 1 main result."
    )
    presenter_notes("BBC 的角色很简单：supplement，不是 main figure。", show_notes)

elif page == PAGES[6]:
    st.markdown("## Dataset 5: ERCE (planned validation)")
    st.markdown("### Why this dataset")
    st.markdown(
        "ERCE political_classifier was intended as a more semantically aligned political-vs-non-political validation set."
    )
    st.markdown("### What happened")
    st.markdown(
        """
- planned as a Step 1 semantic validation
- raw English manually labeled CSV was not available locally
- therefore it remained blocked in this round
"""
    )
    st.info(
        "Interpretation: ERCE remains a useful future validation direction, but it did not become part of the completed result set in this cycle."
    )
    presenter_notes("这一页很短。目的只是交代：我考虑过更语义严格的验证集，但这轮没有形成完整结果。", show_notes)

elif page == PAGES[7]:
    st.markdown("## Method Pivot: Linear Probe + Permutation")
    st.markdown(
        """
At this point, even the best public Step 1 setting only produced angular gaps of a few degrees.
So I asked a methodological question:

> **Is the signal genuinely weak, or is cosine similarity simply not sensitive enough?**
"""
    )

    st.markdown("### Adding a layer-wise logistic regression probe")
    show_image(
        STEP1_LIFESTYLE / "comparison" / "five_models_probe.png",
        "Five models: linear probe performance on HuffPost Lifestyle",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Angular-gap scale")
        st.markdown(
            """
| Model | Peak angular gap |
|-------|------------------|
| Qwen2.5-7B-Instruct | 4.156 deg |
| Qwen2.5-7B | 3.024 deg |
| Mistral-7B-v0.1 | 2.626 deg |
| Qwen2.5-1.5B-Instruct | 1.294 deg |
"""
        )
    with c2:
        st.markdown("#### Probe scale")
        st.markdown(
            """
| Model | Peak accuracy | Peak AUROC |
|-------|---------------|------------|
| Qwen2.5-7B | **97.5%** | 0.993 |
| Mistral-7B-v0.1 | **97.5%** | 0.999 |
| Qwen2.5-7B-Instruct | **97.0%** | 0.993 |
| Qwen2.5-1.5B-Instruct | **94.0%** | 0.984 |
"""
        )

    st.success(
        "This is the key methodological pivot: the signal is not weak. It is strongly linearly decodable, while cosine geometry is noticeably less sensitive."
    )

    st.markdown("### Adding permutation tests")
    st.markdown(
        """
- labels shuffled 200 times
- probe rerun on the shuffled labels
- observed probe performance compared against the null distribution
"""
    )
    st.markdown(
        """
**Focused permutation result:** all five models reached empirical **p < 0.005** on the relevant probe / geometry layers.
"""
    )
    st.warning(
        "Interpretation: the signal is clearly non-random. However, permutation tests do not rule out topic or source confounds; they only show that the observed separation is not due to chance label assignment."
    )
    presenter_notes(
        "这是整个 Step 1 最重要的方法转折页。\n\n"
        "我建议你强调：\n"
        "1. cosine 不是错，只是不够灵敏\n"
        "2. probe 给了主证据\n"
        "3. permutation 给了统计保障",
        show_notes,
    )

elif page == PAGES[8]:
    st.markdown("## Step 1 Integration: Candidate Political Layers")
    st.markdown(
        """
I did not define political layers using geometry alone or probing alone.
Instead, I combined both:

- top-5 angular-gap layers
- top-5 probe AUROC layers
- overlap plus neighboring layers
"""
    )
    show_image(
        STEP1_LIFESTYLE / "comparison" / "five_models_candidate_layers_heatmap.png",
        "Candidate political layers across five models",
    )
    st.markdown(
        """
| Model | Candidate political layers |
|-------|----------------------------|
| Qwen2.5-7B | **22-27** |
| Qwen2.5-1.5B-Instruct | **21-24, 26-27** |
| Qwen2.5-7B-Instruct | **21-27** |
| Mistral-7B-v0.1 | **27-29** |
| Mistral-7B-Instruct-v0.2 | **23-24** |
"""
    )
    st.success(
        """
Step 1 conclusion:

1. political-vs-non-political information is strongly linearly decodable
2. the strongest public proxy result appears in upper-middle to late layers
3. cosine provides a weaker but consistent geometric signature
4. the output is best described as **candidate political layers**, not a hard boundary
"""
    )
    st.warning(
        "This is still a proxy politicality result. It supports a candidate-layer story, but it does not by itself prove a fully confound-free notion of politicality."
    )
    presenter_notes(
        "到这里，Step 1 就可以收束了。\n\n"
        "最稳的说法是 candidate political layers，而不是唯一硬边界。",
        show_notes,
    )

elif page == PAGES[9]:
    st.markdown("## Dataset 6: Article-Bias-Prediction")
    st.markdown("### Why this dataset")
    st.markdown(
        "After Step 1, I moved to ideology analysis using a public left / center / right dataset."
    )
    st.markdown("### Method used")
    st.markdown(
        """
- 100 left + 100 center + 100 right
- title-only text
- 3-class logistic regression
- centroid geometry and center projection
"""
    )
    show_image(
        STEP2_AB / "five_models_public_step2_probe.png",
        "Five models: Article-Bias public Step 2 probe",
    )
    st.markdown(
        """
| Model | Peak AUROC | Peak layer | Step 1 candidate band |
|-------|------------|------------|-----------------------|
| Mistral-7B-v0.1 | **0.914** | 12 | 27-29 |
| Mistral-7B-Instruct-v0.2 | **0.902** | 6 | 23-24 |
| Qwen2.5-7B | **0.894** | 11 | 22-27 |
| Qwen2.5-7B-Instruct | **0.887** | 11 | 21-27 |
| Qwen2.5-1.5B-Instruct | **0.858** | 4 | 21-24, 26-27 |
"""
    )
    st.warning(
        "Critical observation: ideology probe peaks much earlier than the Step 1 candidate political layers. This suggests that the strongest Step 2 signal may reflect a different kind of information."
    )
    st.markdown("### Why this result needed further checking")
    st.markdown(
        """
The source distributions are strongly separated across labels:

- left: ABC, Salon, The Guardian, CNN
- center: BBC, AP, Reuters
- right: CBN, Newsmax, Breitbart, Daily Caller
"""
    )
    st.warning(
        "Interpretation: the strong Step 2 performance on Article-Bias appears to be heavily influenced by source/style confounds. It cannot be interpreted as a clean ideology result on its own."
    )
    presenter_notes(
        "这一页最关键的作用是引出 AllSides。\n\n"
        "不是说 Article-Bias 没价值，而是说它的高分不能直接当 ideology 的干净证据。",
        show_notes,
    )

elif page == PAGES[10]:
    st.markdown("## Dataset 7: AllSides (topic-matched)")
    st.markdown("### Why this dataset")
    st.markdown(
        """
AllSides provides same-event left / center / right roundups.
This makes it a much stricter test, because topic is already controlled at the event level.

This is the key question:

> If the ideology signal is real, does it survive under topic matching?
"""
    )
    st.markdown("### Method used")
    st.markdown(
        """
- 100 event-matched left / center / right triplets
- heading-only text
- same 3-class probe pipeline
- 3,979 complete triplets available overall
"""
    )
    show_image(
        STEP2_ALLSIDES / "five_models_public_step2_probe.png",
        "Five models: AllSides public Step 2 probe",
    )
    st.markdown(
        """
| Model | Article-Bias AUROC | AllSides AUROC | Difference |
|-------|--------------------|----------------|------------|
| Qwen2.5-7B | 0.894 | 0.572 | -0.322 |
| Qwen2.5-1.5B-Instruct | 0.858 | 0.550 | -0.308 |
| Qwen2.5-7B-Instruct | 0.887 | 0.573 | -0.314 |
| Mistral-7B-v0.1 | 0.914 | 0.560 | -0.354 |
| Mistral-7B-Instruct-v0.2 | 0.902 | 0.539 | -0.363 |
"""
    )
    st.error(
        "Under the current pipeline, the ideology signal becomes much weaker on AllSides. This strongly suggests that the earlier Article-Bias result was substantially inflated by source/style differences."
    )
    st.warning(
        "Careful interpretation: this does not prove that ideology is absent. It shows that ideology is not strongly linearly separable under the current AllSides setting and representation pipeline."
    )
    presenter_notes(
        "这是最重要的 Step 2 结果。\n\n"
        "它的意义不是 'AllSides 失败了'，而是它揭示了 Article-Bias 的 confound 结构。",
        show_notes,
    )

elif page == PAGES[11]:
    st.markdown("## AllSides Route B: Ablation")
    st.markdown("### Why this ablation")
    st.markdown(
        "Before concluding too much from AllSides, I checked whether the weak signal was caused by the representation choice rather than the dataset itself."
    )
    st.markdown("### Single-model pilot: Qwen2.5-7B-Instruct")
    st.markdown(
        """
| Config | Text | Pooling | Peak AUROC |
|--------|------|---------|------------|
| heading + last-token | headline | last token | 0.573 |
| heading + mean-pool | headline | mean pool | 0.477 |
| full-text + last-token | full article | last token | **0.587** |
| full-text + mean-pool | full article | mean pool | 0.456 |
"""
    )
    st.info(
        "Result: full-text + last-token gives only a very small improvement. Mean pooling is consistently worse."
    )
    st.warning(
        "Interpretation: the weak AllSides signal is not fixed by simple input or pooling changes. If AllSides is to be pushed further, it will likely require stronger methods, such as nonlinear probes or RFM-style approaches."
    )
    presenter_notes(
        "这页的作用是排除一个 easy explanation：\n"
        "不是因为 headline 太短，也不是因为 last-token 太差。\n"
        "如果继续挖 AllSides，就要换更强方法。",
        show_notes,
    )

else:
    st.markdown("## 13. Integrated Takeaways")
    st.markdown("### Overall picture across datasets")
    st.markdown(
        """
| Dataset | Question | Main signal | Main lesson |
|---------|----------|-------------|-------------|
| Custom matched | politicality | looked strong, probe unstable | small-sample probing was unreliable |
| political_or_not | politicality | weak | data definition mattered more than pairing |
| HuffPost Lifestyle | politicality | **strong** | best current public Step 1 proxy |
| BBC | politicality | moderate | useful supplement, not main result |
| Article-Bias | ideology | **high under current probe** | heavily confounded by source/style |
| AllSides | ideology | weak under current probe | stricter benchmark, much harder under topic control |
"""
    )

    st.markdown("### Three current conclusions")
    st.success(
        """
**1. Step 1 is usable.**
Political-vs-non-political information is strongly linearly decodable, and the strongest public proxy result comes from HuffPost Lifestyle plus headline-only inputs.
"""
    )
    st.info(
        """
**2. Step 2 is more fragile.**
Public ideology decoding works well on Article-Bias-Prediction, but this result appears to be strongly influenced by source/style confounds.
"""
    )
    st.warning(
        """
**3. Politicality and ideology should not be treated as the same representational problem.**
Step 1 candidate political layers and Step 2 ideology peaks do not align cleanly, and the strict AllSides test suggests that ideology is much harder to recover under topic control.
"""
    )

    st.markdown("### Discussion: where to go next")
    st.markdown(
        """
**Option A: write up the current findings as a methodological contribution**
- strong Step 1 result
- cautionary Step 2 result
- confound-sensitive interpretation

**Option B: continue on AllSides with stronger methods**
- nonlinear probes
- RFM-style approaches
- richer representation readouts

**Option C: build a more controlled matched corpus**
- left / right / neutral / non-political
- topic-matched by construction
- one dataset serving both Step 1 and Step 2
"""
    )
    presenter_notes(
        "最后一页不要再堆细节。\n\n"
        "你只需要讲清楚三件事：\n"
        "1. Step 1 已经成立\n"
        "2. Step 2 当前公开结果有 confound\n"
        "3. 接下来是写、换方法、还是做新数据",
        show_notes,
    )


st.markdown("---")
st.caption(
    "This app summarizes results from `/Users/gengliu/Documents/Playground/political-bias-representation-engineering/step1_politicality_v2` "
    "and `/Users/gengliu/Documents/Playground/political-bias-representation-engineering/step2_ideology_public_v1`."
)
