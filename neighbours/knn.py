import numpy as np
from kd_tree import KDNode, KDTree


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
            flag = x[axis] < node_value[axis]
            min_node, min_dist = search_tree(x, node.left_child, depth+1) if flag else \
                search_tree(x, node.right_child, depth+1)
            
            if dist < min_dist:
                # if dist < min_dist, then search another branch
                min_node, min_dist = node, dist
                node, dist = search_tree(x, node.right_child, depth+1) if flag else \
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
        classes = []
        """Using for loop is faster and lower-cost than compute all distance at once"""
        for i in range(X.shape[0]):
            dist = ((X[i] - self.data) ** 2).sum(-1) ** 0.5
            index = np.argsort(dist)[: self.n_neighbors]
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
    # print(cost_naive / cost_kd)
    