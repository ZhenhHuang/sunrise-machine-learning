import numpy as np


class KDNode:
    def __init__(self, node_value=None, left_child=None, right_child=None):
        self.node_value = node_value
        self.left_child = left_child
        self.right_child = right_child
        
    def __str__(self) -> str:
        return str(self.node_value)


class KDTree:
    def __init__(self):
        self.root = KDNode()
        
    def fit(self, X, y=None):
        self.root = self._createTree(X, y)
        return self

    def _createTree(self, points: np.ndarray, labels=None, depth: int = 0):
        if len(points) == 0:
            return None
        axis = depth % points.shape[-1]
        index = np.argsort(points[:, axis])
        points = points[index]     # sorted point and find median
        median = len(points) // 2
        value = {}
        value['data'] = points[median]
        l_labels, r_labels = None, None
        if labels is not None:
            labels = labels[index]
            value['label'] = labels[median]
            l_labels = labels[:median]
            r_labels = labels[median+1:]
        
        return KDNode(
            node_value=value,
            left_child=self._createTree(points[:median], l_labels, depth+1),
            right_child=self._createTree(points[median+1:], r_labels, depth+1)
        )
    
    def __repr__(self) -> str:
        def get_tree_dict(node: KDNode):
            if node is None:
                return None
            if node.left_child is None and node.right_child is None:
                return node.node_value
            node_dict = {str(node):{}}
            node_dict[str(node)]['l'] = get_tree_dict(node.left_child)
            node_dict[str(node)]['r'] = get_tree_dict(node.right_child)
            return node_dict
        return str(get_tree_dict(self.root))