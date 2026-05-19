"""Tests for tree_database module."""

import unittest

from openevolve.config import DatabaseConfig

from opentreesearch.tree_database import TreeNode, TreeProgramDatabase


def make_node(id, parent_id=None, depth=0, visits=1, score=0.5):
    """Create a TreeNode for testing."""
    return TreeNode(
        id=id,
        code=f"# code for {id}",
        iteration_found=0,
        parent_id=parent_id,
        depth=depth,
        visits=visits,
        metrics={"combined_score": score},
    )


class TestTreeNode(unittest.TestCase):
    """Tests for TreeNode dataclass."""

    def test_to_dict_from_dict_roundtrip(self):
        """Verify serialization round-trip preserves fields."""
        node = make_node("n1", parent_id="root", depth=2, visits=3, score=0.75)
        d = node.to_dict()
        restored = TreeNode.from_dict(d)
        self.assertEqual(restored.id, "n1")
        self.assertEqual(restored.parent_id, "root")
        self.assertEqual(restored.depth, 2)
        self.assertEqual(restored.visits, 3)
        self.assertAlmostEqual(restored.metrics["combined_score"], 0.75)

    def test_from_dict_ignores_extra_fields(self):
        """Verify unknown fields are filtered out."""
        d = make_node("n1").to_dict()
        d["unknown_field"] = "should be ignored"
        restored = TreeNode.from_dict(d)
        self.assertEqual(restored.id, "n1")


class TestTreeProgramDatabase(unittest.TestCase):
    """Tests for TreeProgramDatabase."""

    def setUp(self):
        """Set up a fresh database for each test."""
        self.config = DatabaseConfig()
        self.config.db_path = None
        self.db = TreeProgramDatabase(self.config)

    def test_add_root(self):
        """Verify adding a root node."""
        root = make_node("root", score=0.5)
        self.db.add(root, root=True)
        self.assertEqual(len(self.db.tree), 1)
        self.assertEqual(self.db.best_program_id, "root")
        self.assertEqual(self.db.tree["root"].visits, 1)

    def test_add_child_updates_parent(self):
        """Verify child is linked to parent."""
        root = make_node("root", score=0.5)
        self.db.add(root, root=True)

        child = make_node("c1", parent_id="root", depth=1, score=0.6)
        self.db.add(child, root=False)

        self.assertIn("c1", self.db.tree["root"].children_ids)
        self.assertEqual(len(self.db.tree), 2)

    def test_backpropagation_increments_ancestor_visits(self):
        """Verify visit counts propagate up the tree."""
        root = make_node("root", score=0.3)
        self.db.add(root, root=True)

        child1 = make_node("c1", parent_id="root", depth=1, score=0.5)
        self.db.add(child1, root=False)

        grandchild = make_node("gc1", parent_id="c1", depth=2, score=0.7)
        self.db.add(grandchild, root=False)

        self.assertEqual(self.db.tree["root"].visits, 3)
        self.assertEqual(self.db.tree["c1"].visits, 2)
        self.assertEqual(self.db.tree["gc1"].visits, 1)

    def test_get_best_program(self):
        """Verify best program is the highest scoring node."""
        root = make_node("root", score=0.3)
        self.db.add(root, root=True)

        child = make_node("c1", parent_id="root", depth=1, score=0.9)
        self.db.add(child, root=False)

        best = self.db.get_best_program()
        self.assertEqual(best.id, "c1")

    def test_ucb_expand_returns_parent_and_inspirations(self):
        """Verify ucb_expand returns a parent and inspiration list."""
        root = make_node("root", score=0.3)
        self.db.add(root, root=True)

        c1 = make_node("c1", parent_id="root", depth=1, score=0.5)
        self.db.add(c1, root=False)

        c2 = make_node("c2", parent_id="root", depth=1, score=0.8)
        self.db.add(c2, root=False)

        parent, inspirations = self.db.ucb_expand(num_inspirations=2)
        self.assertIsInstance(parent, TreeNode)
        self.assertIsInstance(inspirations, list)
        for insp in inspirations:
            self.assertNotEqual(insp.id, parent.id)

    def test_ucb_expand_single_node(self):
        """Verify ucb_expand works with a single node."""
        root = make_node("root", score=0.5)
        self.db.add(root, root=True)

        parent, inspirations = self.db.ucb_expand()
        self.assertEqual(parent.id, "root")
        self.assertEqual(len(inspirations), 0)

    def test_ucb_expand_deterministic(self):
        """Verify ucb_expand is deterministic for the same tree state."""
        root = make_node("root", score=0.3)
        self.db.add(root, root=True)
        for i in range(5):
            self.db.add(
                make_node(f"c{i}", parent_id="root", depth=1, score=0.1 * (i + 1)),
                root=False,
            )

        p1, _ = self.db.ucb_expand()
        p2, _ = self.db.ucb_expand()
        self.assertEqual(p1.id, p2.id)

    def test_programs_property_aliases_tree(self):
        """Verify the programs property returns the same dict as tree."""
        root = make_node("root", score=0.5)
        self.db.add(root, root=True)
        self.assertIs(self.db.programs, self.db.tree)


class TestTreeNodeCompatibility(unittest.TestCase):
    """Tests for OpenEvolve compatibility stubs."""

    def test_generation_aliases_depth(self):
        """Verify generation property reads and writes depth."""
        node = make_node("n1", depth=3)
        self.assertEqual(node.generation, 3)


if __name__ == "__main__":
    unittest.main()
