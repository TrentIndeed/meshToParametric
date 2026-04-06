from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from mesh2param.models.feature import Feature

logger = logging.getLogger(__name__)


@dataclass
class FeatureTree:
    """An ordered list of features forming a CAD feature tree.

    Output of Stage 5: Feature Recognition.
    Used as input to Stage 6: Onshape Export.
    """

    features: list[Feature] = field(default_factory=list)
    dependency_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        logger.debug(
            "FeatureTree created: %d features, graph nodes=%d, edges=%d",
            len(self.features),
            self.dependency_graph.number_of_nodes(),
            self.dependency_graph.number_of_edges(),
        )
        if len(self.features) == 0:
            logger.warning("FeatureTree created with zero features")

    @property
    def num_features(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict:
        try:
            return {
                "num_features": self.num_features,
                "features": [f.to_dict() for f in self.features],
                "metadata": self.metadata,
            }
        except Exception:
            logger.exception("FeatureTree.to_dict failed (%d features)", len(self.features))
            raise
