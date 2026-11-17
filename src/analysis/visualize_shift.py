
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analysis.shift_analysis import extract_features


DOMAIN_NAMES = ["uci", "wisdm", "pamap2_hand"]


def load_domains(data_dir):
    """Load all available processed HAR datasets."""
    domains = {}

    for name in DOMAIN_NAMES:
        path = Path(data_dir) / f"{name}.npz"

        if not path.exists():
            print(f"Skipping {name}: file not found.")
            continue

        with np.load(path, allow_pickle=False) as data:
            domains[name] = (
                data["X"].copy(),
                data["y"].copy(),
            )

    if len(domains) < 2:
        raise RuntimeError("At least two datasets are required.")

    return domains


def prepare_features(domains, max_samples=3000, seed=0):
    """Extract and standardize features from each domain."""
    rng = np.random.default_rng(seed)

    feature_parts = []
    domain_labels = []
    activity_labels = []

    for domain_id, (name, (X, y)) in enumerate(domains.items()):
        features = extract_features(X)

        if len(features) > max_samples:
            indices = rng.choice(
                len(features), max_samples, replace=False
            )
            features = features[indices]
            y = y[indices]

        feature_parts.append(features)
        domain_labels.extend([name] * len(features))
        activity_labels.extend(y.tolist())

    combined = np.vstack(feature_parts)

    scaler = StandardScaler()
    combined = scaler.fit_transform(combined)

    return (
        combined,
        np.asarray(domain_labels),
        np.asarray(activity_labels),
    )


def calculate_silhouette(embedding, labels):
    """Calculate silhouette score when enough groups are present."""
    unique = np.unique(labels)

    if len(unique) < 2 or len(unique) >= len(labels):
        return None

    return float(silhouette_score(embedding, labels))


def plot_embedding(embedding, labels, title, output_path):
    """Save a 2D scatter plot colored by group."""
    plt.figure(figsize=(9, 7))

    for label in np.unique(labels):
        mask = labels == label

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=str(label),
            alpha=0.55,
            s=12,
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_visualizations(
    domains,
    output_dir="results/figures",
    max_samples=3000,
    seed=0,
):
    features, domain_labels, activity_labels = prepare_features(
        domains, max_samples, seed
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PCA
    pca = PCA(n_components=2, random_state=seed)
    pca_embedding = pca.fit_transform(features)

    plot_embedding(
        pca_embedding,
        domain_labels,
        "PCA - Dataset Distribution",
        output_dir / "pca_by_domain.png",
    )

    plot_embedding(
        pca_embedding,
        activity_labels,
        "PCA - Activity Distribution",
        output_dir / "pca_by_activity.png",
    )

    pca_domain_score = calculate_silhouette(
        pca_embedding, domain_labels
    )
    pca_activity_score = calculate_silhouette(
        pca_embedding, activity_labels
    )

    # t-SNE
    perplexity = min(30, max(2, (len(features) - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )

    tsne_embedding = tsne.fit_transform(features)

    plot_embedding(
        tsne_embedding,
        domain_labels,
        "t-SNE - Dataset Distribution",
        output_dir / "tsne_by_domain.png",
    )

    plot_embedding(
        tsne_embedding,
        activity_labels,
        "t-SNE - Activity Distribution",
        output_dir / "tsne_by_activity.png",
    )

    tsne_domain_score = calculate_silhouette(
        tsne_embedding, domain_labels
    )
    tsne_activity_score = calculate_silhouette(
        tsne_embedding, activity_labels
    )

    scores = {
        "pca_domain_silhouette": pca_domain_score,
        "pca_activity_silhouette": pca_activity_score,
        "tsne_domain_silhouette": tsne_domain_score,
        "tsne_activity_silhouette": tsne_activity_score,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }

    with (output_dir / "visualization_scores.txt").open("w") as file:
        for name, value in scores.items():
            file.write(f"{name}: {value}\n")

    print("Visualizations saved to:", output_dir)
    print(scores)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    domains = load_domains(args.data_dir)

    generate_visualizations(
        domains,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
