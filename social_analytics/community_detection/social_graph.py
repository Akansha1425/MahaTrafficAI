"""Social Graph Analysis — Phase 4, MahaTraffic AI.

REAL DATA CONSTRAINT:
    The dataset (maharashtra_road_safety_public_posts.csv) does NOT contain
    author/user identifiers, mention/reply relationships, or interaction edges.
    Therefore, graph-based community detection CANNOT be applied to the real data.

    This module:
      1. Documents WHY real graph analysis is not possible.
      2. Provides a CLEARLY LABELLED academic demonstration using synthetic
         node/edge data to illustrate the concepts.

    The demonstration is NOT derived from real social network data.
    It is provided purely for academic illustration of the methodology.

Academic Context:
    Graph-based social network analysis (SNA) typically uses:
      - Nodes: users/accounts
      - Edges: mentions, replies, retweets, follows
    This dataset has none of these fields.
"""

from __future__ import annotations
import logging

logger = logging.getLogger("social.graph")

# ─── Availability check ───────────────────────────────────────────────────────

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logger.warning("NetworkX not installed. Graph demo unavailable.")

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.info("python-louvain not installed. Louvain community detection unavailable.")

# ─── Why real graph analysis is not applicable ────────────────────────────────

REAL_DATA_LIMITATION = """
GRAPH ANALYSIS STATUS: NOT APPLICABLE TO REAL DATA
===================================================
Dataset: maharashtra_road_safety_public_posts.csv (1,200 posts)

REASON: The dataset does not contain:
  - Author/user identifiers
  - Mention/reply relationships
  - Retweet/share networks
  - Follower/following information
  - Interaction edges between users

WITHOUT these fields, it is not possible to construct a meaningful
social network graph from the real data.

Fabricating network structure from available data would be methodologically
invalid and is NOT done in this project.

WHAT IS AVAILABLE (structural signals only):
  - Location co-occurrence (multiple posts from same district)
  - Hashtag co-occurrence (posts sharing hashtags)
  - Category clustering (posts grouped by topic)

These cannot substitute for genuine social network topology.

ACADEMIC DEMONSTRATION:
  The run_academic_demo() function below illustrates the algorithms
  using a small synthetic example. This is clearly labelled as a demo.
"""


def get_limitation_statement() -> str:
    """Return the documented limitation statement for graph analysis."""
    return REAL_DATA_LIMITATION


# ─── Academic demonstration (synthetic data) ──────────────────────────────────

def run_academic_demo() -> dict | None:
    """
    [ACADEMIC DEMONSTRATION — NOT REAL DATA]

    Illustrates graph analytics concepts using a synthetic 10-node network
    representing hypothetical social interactions between road-safety advocates.

    This demonstration is provided for academic completeness only.
    Results do NOT represent any real users or real social interactions.
    """
    if not NX_AVAILABLE:
        print("[DEMO SKIPPED] NetworkX not installed.")
        return None

    print("\n" + "=" * 70)
    print("[ACADEMIC DEMONSTRATION — SYNTHETIC DATA — NOT REAL USERS]")
    print("Road-Safety Advocate Network — Illustrative Example")
    print("=" * 70)

    # Synthetic nodes (hypothetical road-safety advocates / NGO accounts)
    nodes = [
        "SafeRoadsNGO", "MahaTrafficWatch", "PotholeReporter",
        "PuneRoadSafety", "MumbaiCommuter", "NashikCivic",
        "PedestrianFirst", "HighwayWatcher", "DrunkDrivingAware", "SpeedCampaign",
    ]

    # Synthetic edges (hypothetical interactions)
    edges = [
        ("SafeRoadsNGO", "MahaTrafficWatch"),
        ("SafeRoadsNGO", "PotholeReporter"),
        ("MahaTrafficWatch", "PuneRoadSafety"),
        ("MahaTrafficWatch", "MumbaiCommuter"),
        ("PotholeReporter", "NashikCivic"),
        ("PuneRoadSafety", "PedestrianFirst"),
        ("MumbaiCommuter", "PedestrianFirst"),
        ("NashikCivic", "HighwayWatcher"),
        ("HighwayWatcher", "DrunkDrivingAware"),
        ("DrunkDrivingAware", "SpeedCampaign"),
        ("SpeedCampaign", "SafeRoadsNGO"),
        ("PedestrianFirst", "SpeedCampaign"),
    ]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Connected: {nx.is_connected(G)}")
    print(f"  Components: {nx.number_connected_components(G)}")

    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    top_degree = sorted(degree_centrality.items(), key=lambda x: -x[1])[:5]
    print("\n  Top 5 by Degree Centrality:")
    for node, score in top_degree:
        print(f"    {node:<25} {score:.4f}")

    # PageRank
    pagerank = nx.pagerank(G, alpha=0.85)
    top_pr = sorted(pagerank.items(), key=lambda x: -x[1])[:5]
    print("\n  Top 5 by PageRank:")
    for node, score in top_pr:
        print(f"    {node:<25} {score:.4f}")

    # Community detection
    communities = {}
    if LOUVAIN_AVAILABLE:
        communities = community_louvain.best_partition(G)
        community_groups: dict[int, list] = {}
        for node, comm_id in communities.items():
            community_groups.setdefault(comm_id, []).append(node)
        print("\n  Communities (Louvain):")
        for cid, members in community_groups.items():
            print(f"    Community {cid}: {members}")
    else:
        # Fallback: greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities
        comm_sets = list(greedy_modularity_communities(G))
        print("\n  Communities (Greedy Modularity):")
        for i, c in enumerate(comm_sets):
            print(f"    Community {i}: {sorted(c)}")
        for i, c in enumerate(comm_sets):
            for node in c:
                communities[node] = i

    # Summary dict
    result = {
        "data_type": "SYNTHETIC — ACADEMIC DEMONSTRATION ONLY",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "is_connected": nx.is_connected(G),
        "num_components": nx.number_connected_components(G),
        "degree_centrality": {k: round(v, 4) for k, v in degree_centrality.items()},
        "pagerank": {k: round(v, 4) for k, v in pagerank.items()},
        "community_assignment": communities,
    }

    print("\n" + "=" * 70)
    print("[END ACADEMIC DEMONSTRATION]")
    print("=" * 70 + "\n")

    return result


def run_social_graph_analysis() -> dict | None:
    """Run academic demonstration of graph analysis."""
    return run_academic_demo()


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(REAL_DATA_LIMITATION)
    demo_result = run_academic_demo()
    if demo_result:
        print("Demo complete. Results are SYNTHETIC — not from real data.")
