"""
Institutional Graph Store for organizational knowledge.

This module provides a graph-based layer for representing and querying
organizational structures and relationships that are better modeled as graphs.

Use cases:
- Organizational hierarchy (reporting chains, teams, departments)
- Customer relationships (accounts → contacts → interactions)
- Product dependencies (components → assemblies → products)
- Document references (policies → procedures → forms)

Storage: Uses local JSON files in `data/graph/` for persistence.
Future: Can be upgraded to Neo4j or similar for larger scale.

Status: STUB - Interface defined, full implementation planned for future phase.
"""

import json
import os
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class GraphNode:
    """A node in the institutional graph."""
    id: str
    type: str  # 'person', 'team', 'document', 'product', etc.
    name: str
    properties: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GraphEdge:
    """An edge (relationship) in the institutional graph."""
    source_id: str
    target_id: str
    relationship: str  # 'reports_to', 'owns', 'references', 'depends_on', etc.
    properties: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GraphQueryResult:
    """Result from a graph query."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)  # Node ID sequences


class InstitutionalGraph:
    """
    Graph store for organizational knowledge and relationships.

    Provides:
    - Node and edge CRUD operations
    - Path queries (e.g., "Who does Alice report to?")
    - Subgraph extraction (e.g., "Get all of Bob's team")
    - Integration with UnifiedEngine for hybrid queries

    Example usage:
        graph = InstitutionalGraph()
        graph.add_node(GraphNode(id='alice', type='person', name='Alice Smith'))
        graph.add_node(GraphNode(id='bob', type='person', name='Bob Jones'))
        graph.add_edge(GraphEdge(source_id='alice', target_id='bob', relationship='reports_to'))

        # Query: Who does Alice report to?
        result = graph.traverse('alice', 'reports_to', direction='outgoing')
    """

    # Supported node types
    NODE_TYPES = {'person', 'team', 'department', 'document', 'product', 'customer', 'account'}

    # Supported relationship types
    RELATIONSHIP_TYPES = {
        'reports_to', 'manages', 'belongs_to', 'owns',
        'references', 'depends_on', 'related_to', 'contacts'
    }

    def __init__(self, storage_path: str = "data/graph"):
        """
        Initialize the graph store.

        Args:
            storage_path: Directory for graph data persistence
        """
        self.storage_path = Path(storage_path)
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)

        # Load existing data if present
        self._load()

    def _load(self):
        """Load graph data from storage."""
        nodes_file = self.storage_path / "nodes.json"
        edges_file = self.storage_path / "edges.json"

        if nodes_file.exists():
            try:
                with open(nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                    self.nodes = {
                        n['id']: GraphNode(**n) for n in nodes_data
                    }
            except (json.JSONDecodeError, KeyError):
                self.nodes = {}

        if edges_file.exists():
            try:
                with open(edges_file, 'r') as f:
                    edges_data = json.load(f)
                    self.edges = [GraphEdge(**e) for e in edges_data]
            except (json.JSONDecodeError, KeyError):
                self.edges = []

    def _save(self):
        """Persist graph data to storage."""
        nodes_file = self.storage_path / "nodes.json"
        edges_file = self.storage_path / "edges.json"

        with open(nodes_file, 'w') as f:
            json.dump([asdict(n) for n in self.nodes.values()], f, indent=2)

        with open(edges_file, 'w') as f:
            json.dump([asdict(e) for e in self.edges], f, indent=2)

    def add_node(self, node: GraphNode) -> bool:
        """
        Add a node to the graph.

        Args:
            node: GraphNode to add

        Returns:
            True if added, False if already exists
        """
        if node.id in self.nodes:
            return False

        self.nodes[node.id] = node
        self._save()
        return True

    def add_edge(self, edge: GraphEdge) -> bool:
        """
        Add an edge to the graph.

        Args:
            edge: GraphEdge to add

        Returns:
            True if added, False if nodes don't exist
        """
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return False

        self.edges.append(edge)
        self._save()
        return True

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.type == node_type]

    def traverse(
        self,
        start_id: str,
        relationship: Optional[str] = None,
        direction: str = 'both',
        max_depth: int = 1
    ) -> GraphQueryResult:
        """
        Traverse the graph from a starting node.

        Args:
            start_id: Starting node ID
            relationship: Optional relationship type to filter
            direction: 'outgoing', 'incoming', or 'both'
            max_depth: Maximum traversal depth

        Returns:
            GraphQueryResult with found nodes and edges
        """
        if start_id not in self.nodes:
            return GraphQueryResult()

        found_nodes = {start_id: self.nodes[start_id]}
        found_edges = []
        to_visit = [(start_id, 0)]
        visited = set()

        while to_visit:
            current_id, depth = to_visit.pop(0)
            if current_id in visited or depth >= max_depth:
                continue
            visited.add(current_id)

            for edge in self.edges:
                # Check direction
                if direction in ('outgoing', 'both') and edge.source_id == current_id:
                    if relationship is None or edge.relationship == relationship:
                        found_edges.append(edge)
                        if edge.target_id not in found_nodes:
                            found_nodes[edge.target_id] = self.nodes[edge.target_id]
                            to_visit.append((edge.target_id, depth + 1))

                if direction in ('incoming', 'both') and edge.target_id == current_id:
                    if relationship is None or edge.relationship == relationship:
                        found_edges.append(edge)
                        if edge.source_id not in found_nodes:
                            found_nodes[edge.source_id] = self.nodes[edge.source_id]
                            to_visit.append((edge.source_id, depth + 1))

        return GraphQueryResult(
            nodes=list(found_nodes.values()),
            edges=found_edges
        )

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[list[str]]:
        """
        Find a path between two nodes.

        Args:
            start_id: Starting node ID
            end_id: Target node ID
            max_depth: Maximum path length

        Returns:
            List of node IDs representing the path, or None if no path found
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        # BFS for shortest path
        queue = [(start_id, [start_id])]
        visited = set()

        while queue:
            current_id, path = queue.pop(0)

            if current_id == end_id:
                return path

            if current_id in visited or len(path) > max_depth:
                continue
            visited.add(current_id)

            # Get neighbors
            for edge in self.edges:
                next_id = None
                if edge.source_id == current_id:
                    next_id = edge.target_id
                elif edge.target_id == current_id:
                    next_id = edge.source_id

                if next_id and next_id not in visited:
                    queue.append((next_id, path + [next_id]))

        return None

    def query_natural_language(self, question: str) -> GraphQueryResult:
        """
        Query the graph using natural language.

        This is a STUB - full implementation would use LLM to parse
        the question and generate appropriate graph traversals.

        Args:
            question: Natural language question

        Returns:
            GraphQueryResult (empty in stub)
        """
        # TODO: Implement NL → Graph query translation
        # Example questions:
        # - "Who does Alice report to?"
        # - "Show me Bob's team structure"
        # - "What documents reference the HR policy?"
        return GraphQueryResult()

    def get_context_for_query(self, entity_names: list[str]) -> str:
        """
        Get graph context for entities mentioned in a query.

        Used by UnifiedEngine to augment retrieval with graph knowledge.

        Args:
            entity_names: List of entity names to look up

        Returns:
            Formatted context string
        """
        context_parts = []

        for name in entity_names:
            # Find nodes matching the name
            matches = [n for n in self.nodes.values()
                      if name.lower() in n.name.lower()]

            for node in matches:
                result = self.traverse(node.id, max_depth=1)
                if result.nodes:
                    context_parts.append(f"\n{node.name} ({node.type}):")
                    for edge in result.edges:
                        if edge.source_id == node.id:
                            target = self.nodes.get(edge.target_id)
                            if target:
                                context_parts.append(
                                    f"  → {edge.relationship} → {target.name}"
                                )
                        else:
                            source = self.nodes.get(edge.source_id)
                            if source:
                                context_parts.append(
                                    f"  ← {edge.relationship} ← {source.name}"
                                )

        return '\n'.join(context_parts) if context_parts else ""

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": list(set(n.type for n in self.nodes.values())),
            "relationship_types": list(set(e.relationship for e in self.edges))
        }

    def clear(self):
        """Clear all nodes and edges."""
        self.nodes = {}
        self.edges = []
        self._save()


if __name__ == "__main__":
    # Quick demo
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        graph = InstitutionalGraph(storage_path=tmpdir)

        # Create org structure
        graph.add_node(GraphNode(id='ceo', type='person', name='CEO Jane'))
        graph.add_node(GraphNode(id='vp_eng', type='person', name='VP Engineering Bob'))
        graph.add_node(GraphNode(id='dev1', type='person', name='Developer Alice'))
        graph.add_node(GraphNode(id='dev2', type='person', name='Developer Charlie'))
        graph.add_node(GraphNode(id='eng_team', type='team', name='Engineering Team'))

        graph.add_edge(GraphEdge(source_id='vp_eng', target_id='ceo', relationship='reports_to'))
        graph.add_edge(GraphEdge(source_id='dev1', target_id='vp_eng', relationship='reports_to'))
        graph.add_edge(GraphEdge(source_id='dev2', target_id='vp_eng', relationship='reports_to'))
        graph.add_edge(GraphEdge(source_id='dev1', target_id='eng_team', relationship='belongs_to'))
        graph.add_edge(GraphEdge(source_id='dev2', target_id='eng_team', relationship='belongs_to'))

        print("Graph Stats:", graph.get_stats())

        # Query: Who does Alice report to?
        result = graph.traverse('dev1', 'reports_to', direction='outgoing')
        print("\nAlice reports to:", [n.name for n in result.nodes if n.id != 'dev1'])

        # Query: Find path from Developer to CEO
        path = graph.find_path('dev1', 'ceo')
        print("\nPath from Alice to CEO:", path)

        # Get context for query
        context = graph.get_context_for_query(['Alice', 'Bob'])
        print("\nGraph context:", context)
