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

    
    
class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, algorithm='default'):
        """_summary_

        Args:
            n_neighbors (_type_): number of nearest neighbors.
            algorithm (str, ["default", "kd_tree"]): algorithm to fit and predict. Defaults to 'default'.
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        
    def fit(self, X, y):
        if self.algorithm == 'default':
            self.__fit_default(X, y)
        elif self.algorithm == 'kd_tree':
            self.__fit_kd_tree(X, y)
    
    def predict(self, X):
        if self.algorithm == 'default':
            return self.__predict_default(X)
        elif self.algorithm == 'kd_tree':
            return self.__predict_kd_tree(X)
    
    def __fit_default(self, X, y):
        self.labels = y
        self.data = X
        self.n_classes = y.max() + 1
        
    def __fit_kd_tree(self, X, y):
        self.tree = KDTree().fit(X, y)
    
    def __predict_kd_tree(self, X):
        def search_tree(x, node: KDNode, depth=0):
            if node is None:
                return None, np.Inf

            axis = depth % len(x)
            node_value = node.node_value['data']   # current node
            dist = np.sum((x - node_value) ** 2) ** 0.5
            if node.left_child is None and node.left_child is None:
                return node, dist
            min_node, min_dist = search_tree(x, node.left_child, depth+1) if x[axis] < node_value[axis] else \
                search_tree(x, node.right_child, depth+1)
            
            if dist < min_dist:
                # if dist < min_dist, then search another branch
                min_node, min_dist = node, dist
                node, dist = search_tree(x, node.right_child, depth+1) if x[axis] < node_value[axis] else \
                    search_tree(x, node.left_child, depth+1)
                if dist < min_dist:
                     min_node, min_dist = node, dist
            return min_node, min_dist
        results = []
        for i in range(X.shape[0]):
            results.append(search_tree(X[i], self.tree.root)[0].node_value['label'])
        return results
    
    def __predict_default(self, X):
        """_summary_
        X.shape: M, D
        self.data.shape: N, D
        """
        dist = (X[:, None, :] - self.data[None, :, :]) ** 2     # M, N, D
        dist = dist.sum(axis=-1) ** 0.5
        classes = []
        for i in range(X.shape[0]):
            index = np.argsort(dist[i])[: self.n_neighbors]
            label = self.labels[index]
            count = np.zeros(self.n_classes)
            for j in range(self.n_neighbors):
                count[label[j]] += 1
            classes.append(count.argmax())
        return np.array(classes)


if __name__ == '__main__':
    import time
    point_list = np.array([[7, 2], [5, 4], [9, 6], [4, 7], [8, 1], [2, 3]])
    label_list = np.array([1, 1, 1, 1, 1, 0])
    knn = KNeighborsClassifier(algorithm='kd_tree')
    knn.fit(point_list, label_list)
    time_now = time.time()
    results = knn.predict(np.array([[3, 4.5]]))
    print(results)
    cost_kd = time.time() - time_now
    print(cost_kd)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(point_list, label_list)
    time_now = time.time()
    results = knn.predict(np.array([[3, 4.5]]))
    print(results)
    cost_naive = time.time() - time_now
    print(cost_naive)
    print(cost_naive / cost_kd)
    