from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from mesh2param.models.feature import Feature


@dataclass
class FeatureTree:
    """An ordered list of features forming a CAD feature tree.

    Output of Stage 5: Feature Recognition.
    Used as input to Stage 6: Onshape Export.
    """

    features: list[Feature] = field(default_factory=list)
    dependency_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    metadata: dict = field(default_factory=dict)

    @property
    def num_features(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict:
        return {
            "num_features": self.num_features,
            "features": [f.to_dict() for f in self.features],
            "metadata": self.metadata,
        }
