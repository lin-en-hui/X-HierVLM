"""
Taxonomy tree utilities: loading, querying, and path operations.

The tree is stored as a nested dictionary with the following structure:
{
    "Animalia": {
        "Phylum1": {
            "Class1": {
                "Order1": {
                    "Family1": {
                        "Genus1": ["species_a", "species_b"],
                        "Genus2": ["species_c"]
                    }
                }
            }
        }
    }
}
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import lru_cache

# Global cache for the tree
_TREE_CACHE = None
_TREE_PATH = None


def load_tree(tree_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the taxonomy tree from a JSON file.

    Args:
        tree_path: Path to the JSON file. If None, uses the path from config.

    Returns:
        The tree dictionary.

    Example:
        >>> tree = load_tree()
        >>> tree["Animalia"]["Chordata"]["Aves"]["Passeriformes"]
    """
    global _TREE_CACHE, _TREE_PATH

    # Use cached tree if available and path hasn't changed
    if _TREE_CACHE is not None and _TREE_PATH == tree_path:
        return _TREE_CACHE

    if tree_path is None:
        from config.settings import TREE_JSON
        tree_path = TREE_JSON

    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree JSON not found: {tree_path}")

    with open(tree_path, 'r', encoding='utf-8') as f:
        _TREE_CACHE = json.load(f)
        _TREE_PATH = tree_path

    print(f"[Tree] Loaded taxonomy tree from: {tree_path}")
    return _TREE_CACHE


def get_species_list_from_tree(tree: Optional[Dict] = None) -> List[str]:
    """
    Extract all species (leaf nodes) from the taxonomy tree.

    Args:
        tree: The tree dictionary. If None, loads from default path.

    Returns:
        Sorted list of all species names.

    Example:
        >>> species = get_species_list_from_tree()
        >>> len(species)
        758
    """
    if tree is None:
        tree = load_tree()

    root = tree.get("Animalia", tree)
    species_list = []

    def dfs(node: Union[Dict, List]) -> None:
        if isinstance(node, list):
            species_list.extend(node)
        elif isinstance(node, dict):
            for child in node.values():
                dfs(child)

    dfs(root)
    return sorted(set(species_list))


def get_all_nodes_at_level(
    tree: Dict,
    level: int,
    include_children: bool = True
) -> List[str]:
    """
    Get all node names at a specific depth level.

    Args:
        tree: The tree dictionary.
        level: The depth level (1=L1, 2=L2, ..., 6=L6).
        include_children: If True, includes all nodes up to this level.

    Returns:
        List of node names at the specified level.

    Example:
        >>> nodes = get_all_nodes_at_level(tree, 2)  # All classes
    """
    root = tree.get("Animalia", tree)
    nodes = []

    def dfs(node: Union[Dict, List], depth: int) -> None:
        if depth == level:
            if isinstance(node, dict):
                nodes.extend(node.keys())
            elif isinstance(node, list):
                nodes.extend(node)
            return

        if isinstance(node, dict):
            for child in node.values():
                dfs(child, depth + 1)
        elif isinstance(node, list):
            # If we hit a list before target level, this is a species list
            # which means the tree is shallower than expected
            return

    dfs(root, 1)
    return sorted(set(nodes))


def get_valid_children(
    path: List[str],
    tree: Optional[Dict] = None
) -> List[str]:
    """
    Get the valid child nodes for a given path in the taxonomy tree.

    Args:
        path: List of node names from root to current node.
        tree: The tree dictionary. If None, loads from default path.

    Returns:
        List of child node names (empty if path is invalid or at leaf).

    Example:
        >>> children = get_valid_children(["Animalia", "Chordata", "Aves"])
        >>> # Returns all orders under Aves
    """
    if tree is None:
        tree = load_tree()

    node = tree.get("Animalia", tree)

    for taxon in path:
        if isinstance(node, dict) and taxon in node:
            node = node[taxon]
        elif isinstance(node, list):
            # If we're at a species list, there are no children
            return []
        else:
            # Path doesn't exist
            return []

    if isinstance(node, dict):
        return list(node.keys())
    elif isinstance(node, list):
        return node
    return []


def find_path_for_species(
    species: str,
    tree: Optional[Dict] = None,
    normalize: bool = True
) -> Optional[List[str]]:
    """
    Find the full taxonomic path (L1~L6) for a given species.

    Args:
        species: The species name (e.g., "Cypherotylus californicus").
        tree: The tree dictionary. If None, loads from default path.
        normalize: If True, handles both "Genus species" and "Genus_species" formats.

    Returns:
        List of length 6 with [L1, L2, L3, L4, L5, L6], or None if not found.

    Example:
        >>> path = find_path_for_species("Cypherotylus californicus")
        >>> ['Animalia', 'Arthropoda', 'Insecta', 'Coleoptera', 'Erotylidae', 'Cypherotylus californicus']
    """
    if tree is None:
        tree = load_tree()

    # Normalize species name: handle underscores and extra spaces
    if normalize:
        species_normalized = species.replace('_', ' ').strip()
    else:
        species_normalized = species

    root = tree.get("Animalia", tree)

    def dfs(node: Union[Dict, List], path: List[str]) -> Optional[List[str]]:
        if isinstance(node, list):
            # Check if the species is in this list (with normalization)
            for sp in node:
                sp_normalized = sp.replace('_', ' ').strip()
                if sp_normalized == species_normalized:
                    return path + [sp]  # Keep the original species name from tree
            return None

        if isinstance(node, dict):
            for key, child in node.items():
                result = dfs(child, path + [key])
                if result is not None:
                    return result
        return None

    return dfs(root, [])


def get_ancestors_for_species(
    species: str,
    tree: Optional[Dict] = None
) -> Optional[Dict[str, str]]:
    """
    Get the hierarchical ancestors for a given species.

    Returns a dict with keys L1~L6.

    Example:
        >>> ancestors = get_ancestors_for_species("Cypherotylus californicus")
        >>> {
        ...     "L1": "Animalia",
        ...     "L2": "Arthropoda",
        ...     "L3": "Insecta",
        ...     "L4": "Coleoptera",
        ...     "L5": "Erotylidae",
        ...     "L6": "Cypherotylus californicus"
        ... }
    """
    path = find_path_for_species(species, tree)
    if path is None:
        return None

    ancestors = {}
    for i, name in enumerate(path):
        ancestors[f"L{i+1}"] = name
    return ancestors


def is_valid_path(
    path: List[str],
    tree: Optional[Dict] = None
) -> bool:
    """
    Check if a given path is valid in the taxonomy tree.

    Args:
        path: List of node names from root to leaf.
        tree: The tree dictionary. If None, loads from default path.

    Returns:
        True if the path exists in the tree, False otherwise.
    """
    if tree is None:
        tree = load_tree()

    node = tree.get("Animalia", tree)

    for i, taxon in enumerate(path):
        if isinstance(node, dict):
            if taxon in node:
                node = node[taxon]
            else:
                return False
        elif isinstance(node, list):
            # If we're at a species list, the path should end here
            return i == len(path) - 1 and taxon in node
        else:
            return False

    # If we've consumed the entire path, it's valid
    return True


def get_subtree(
    path: List[str],
    tree: Optional[Dict] = None
) -> Optional[Union[Dict, List]]:
    """
    Get the subtree rooted at the given path.

    Args:
        path: List of node names from root to the desired node.
        tree: The tree dictionary. If None, loads from default path.

    Returns:
        The subtree (dict or list) at the given path, or None if not found.
    """
    if tree is None:
        tree = load_tree()

    node = tree.get("Animalia", tree)

    for taxon in path:
        if isinstance(node, dict) and taxon in node:
            node = node[taxon]
        else:
            return None

    return node


def get_tree_depth(tree: Optional[Dict] = None) -> int:
    """
    Get the maximum depth of the taxonomy tree.

    Args:
        tree: The tree dictionary. If None, loads from default path.

    Returns:
        Maximum depth (number of levels) in the tree.
    """
    if tree is None:
        tree = load_tree()

    def max_depth(node: Union[Dict, List], current_depth: int) -> int:
        if isinstance(node, list):
            return current_depth
        if isinstance(node, dict):
            if not node:
                return current_depth
            return max(max_depth(child, current_depth + 1) for child in node.values())
        return current_depth

    root = tree.get("Animalia", tree)
    return max_depth(root, 1)


def get_species_count(tree: Optional[Dict] = None) -> int:
    """
    Get the total number of species (leaf nodes) in the tree.

    Args:
        tree: The tree dictionary. If None, loads from default path.

    Returns:
        Total number of species.
    """
    if tree is None:
        tree = load_tree()

    count = 0
    root = tree.get("Animalia", tree)

    def dfs(node: Union[Dict, List]) -> None:
        nonlocal count
        if isinstance(node, list):
            count += len(node)
        elif isinstance(node, dict):
            for child in node.values():
                dfs(child)

    dfs(root)
    return count


def get_stats(tree: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get statistics about the taxonomy tree.

    Args:
        tree: The tree dictionary. If None, loads from default path.

    Returns:
        Dict with keys: total_species, depth, nodes_per_level, etc.
    """
    if tree is None:
        tree = load_tree()

    stats = {
        "total_species": get_species_count(tree),
        "depth": get_tree_depth(tree),
        "nodes_per_level": {}
    }

    for level in range(1, 7):
        nodes = get_all_nodes_at_level(tree, level)
        stats["nodes_per_level"][f"L{level}"] = len(nodes)

    return stats


# ============================================================================
# Convenience: cached versions for frequent queries
# ============================================================================

@lru_cache(maxsize=1024)
def _cached_find_path(species: str) -> Optional[tuple]:
    """Cached version of find_path_for_species (returns tuple for hashability)."""
    result = find_path_for_species(species)
    return tuple(result) if result is not None else None


def find_path_cached(species: str) -> Optional[List[str]]:
    """
    Cached version of find_path_for_species.

    Useful when querying many species from a CSV file.
    """
    result = _cached_find_path(species)
    return list(result) if result is not None else None


# ============================================================================
# Debug / visualization
# ============================================================================

def print_tree_summary(tree: Optional[Dict] = None) -> None:
    """Print a summary of the taxonomy tree."""
    if tree is None:
        tree = load_tree()

    stats = get_stats(tree)
    print("=" * 60)
    print("Taxonomy Tree Summary")
    print("=" * 60)
    print(f"Total Species: {stats['total_species']}")
    print(f"Max Depth: {stats['depth']}")
    print("\nNodes per level:")
    for level, count in stats["nodes_per_level"].items():
        print(f"  {level}: {count}")
    print("=" * 60)


def print_path(path: List[str]) -> str:
    """Format a path for human-readable display."""
    return " -> ".join(path) if path else "None"


# ============================================================================
# Example usage (if run directly)
# ============================================================================

if __name__ == "__main__":
    # Load tree
    tree = load_tree()

    # Print stats
    print_tree_summary(tree)

    # Test species lookup
    test_species = "Cypherotylus californicus"
    path = find_path_for_species(test_species, tree)
    print(f"\nPath for {test_species}:")
    print(f"  {print_path(path)}")

    # Test children lookup
    children = get_valid_children(["Animalia", "Arthropoda", "Insecta"], tree)
    print(f"\nFirst 10 orders under Insecta:")
    for order in children[:10]:
        print(f"  - {order}")

    # Test ancestor dict
    ancestors = get_ancestors_for_species(test_species, tree)
    print(f"\nAncestors for {test_species}:")
    for level, name in ancestors.items():
        print(f"  {level}: {name}")