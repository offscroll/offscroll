"""HDBSCAN clustering over embeddings."""

from __future__ import annotations

import logging

import numpy as np

from offscroll.models import FeedItem

logger = logging.getLogger(__name__)


def cluster_items(
    items: list[FeedItem],
    config: dict,
) -> list[FeedItem]:
    """Cluster embedded items using HDBSCAN.

    Groups items by semantic similarity. Each item's cluster_id
    is set in-place. Items without embeddings are skipped.
    Noise points (items that do not fit any cluster) get
    cluster_id = -1.

    Args:
        items: FeedItems with embedding field populated.
        config: The OffScroll config dict. Uses
            config["clustering"]["min_cluster_size"] (default: 3).

    Returns:
        The same list of items with cluster_id populated.
    """
    # Get config
    min_cluster_size = config.get("clustering", {}).get("min_cluster_size", 3)

    # Filter to items with embeddings
    items_with_embeddings = [item for item in items if item.embedding is not None]

    # If too few items, assign all to noise
    if len(items_with_embeddings) < min_cluster_size:
        for item in items:
            if item.embedding is not None:
                item.cluster_id = -1
        logger.info(
            f"Fewer than {min_cluster_size} items with embeddings. "
            "Assigning all to noise cluster (-1)."
        )
        return items

    # Convert embeddings to numpy array
    embedding_matrix = np.array(
        [item.embedding for item in items_with_embeddings],
        dtype=np.float32,
    )

    # Run HDBSCAN
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding_matrix)

    # Assign cluster_id to items with embeddings
    for item, label in zip(items_with_embeddings, labels, strict=True):
        item.cluster_id = int(label)

    # Log results
    unique_labels = set(labels)
    noise_count = sum(1 for label in labels if label == -1)
    cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)

    logger.info(
        f"Clustered {len(items_with_embeddings)} items into "
        f"{cluster_count} clusters with {noise_count} noise points."
    )

    return items
