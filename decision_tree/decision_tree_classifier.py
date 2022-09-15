from textwrap import indent
import numpy as np
from collections import Counter
from utils import calc_Entropy, Calc_ConditonEntropy


class DecisionTreeClassifier:
    def __init__(self, criterion='info_gain'):
        """_summary_

        Args:
            criterion (str): criterion to split: {'info_gain', 'gini'}. Defaults to 'info_gain'.
        """
        self.criterion = criterion
    
    def fit(self, X, y):
        self.feat_label = list(X[0])
        X = X[1:]
        n_samples, features = X.shape
        if y.ndim == 1:
            y = y[:, None]
            
        self.output_dims = y.shape[1]
        self.classes_ = []
        self.n_classes_ = []
        y_encoded = np.zeros_like(y, dtype=int)
        for i in range(self.output_dims):
            class_i, y_encoded[:, i] = np.unique(y[:, i], return_inverse=True)
            self.classes_.append(class_i)
            self.n_classes_.append(class_i.shape[0])
        self.n_classes_ = np.array(self.n_classes_, dtype=int)
        self.tree = self._generateTree(X, y_encoded, self.feat_label)
    
    def _generateTree(self, X, y, feat_label):
        if np.unique(y, axis=0).shape[0] == 1:
            return int(y[0])
        
        if X.shape[1] == 0 or np.unique(X, axis=0).shape[0] == 1:
            return self.get_best(y)
        
        spliter = self._getBestSpliter(X, y)
        count = Counter([a for a in X[:, spliter]])
        tree = {feat_label[spliter]: {}}
        sub_label = feat_label.copy()
        sub_label.remove(feat_label[spliter])
        index = list(range(X.shape[1]))
        index.remove(spliter)
        sub_X = X[:, index]
        for i, feat in enumerate(count.keys()):
            bool_index = X[:, spliter] == feat
            tree[feat_label[spliter]][feat] = self._generateTree(sub_X[bool_index, :], y[bool_index, :], sub_label)
        return tree
    
    def predict(self, x):
        feat_map = {k: v for v, k in enumerate(self.feat_label)}
        result = []
        
        def DSF(x, tree):
            result = -1
            if not isinstance(tree, dict):
                return tree
            for key, value in tree.items():
                index = feat_map[key]
                result = DSF(x, value[str(x[index])])
                if isinstance(result, int):
                    return result
        
        for i in range(len(x)):
            result.append(DSF(x[i], self.tree))
            
        return np.array(result)
    
    def get_best(self, y):
        count_dict = Counter([l for l in y])
        return count_dict.most_common(1)[0][0]
    
    def _getBestSpliter(self, X, y):
        ent_0 = calc_Entropy(y)
        best_ent = 0.
        best = -1
        for i in range(X.shape[-1]):
            gain = ent_0 - Calc_ConditonEntropy(X[:, i], y)
            if gain >= best_ent:
                best_ent = gain
                best = i
        return best


if __name__ == '__main__':
    X = np.array([['no surfacing', 'flippers'], [1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    y = np.array(['yes', 'yes', 'no', 'no', 'no'])
    cls = DecisionTreeClassifier()
    cls.fit(X, y)
    print(cls.tree)
    print(cls.predict(X[1:]))