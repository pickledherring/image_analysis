import math

class KNN_Classifier():
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_pred):
        num_feats = len(self.X[0])
    
        preds = []
        for x in X_pred:
            # k_distances: [distance, index]
            k_distances = [[100000000, i] for i in range(self.k)]
            for i in range(len(self.X)):
                # find euclidean distance
                sq_diffs = [(x[j] - self.X[i][j])**2 for j in range(num_feats)]
                dist = math.sqrt(sum(sq_diffs))
                # compare to current neighbors
                for k in range(len(k_distances)):
                    if dist < k_distances[k][0]:
                        # shift up farther objects in the list
                        for k_r in range(len(k_distances) - 1, k, -1):
                            k_distances[k_r][0] = k_distances[k_r-1][0]
                            k_distances[k_r][1] = k_distances[k_r-1][1]
                        k_distances[k][0] = dist
                        k_distances[k][1] = i
            # evaluate final nearest neighbors            
            classes = {}
            for nn in k_distances:
                if str(self.y[nn[1]]) in classes.keys():
                    classes[str(self.y[nn[1]])] += 1
                else:
                    classes[str(self.y[nn[1]])] = 1

            # find first mode of the neighbors' classes
            max_class = str(self.y[k_distances[0][1]])
            max_value = classes[max_class]
            for key in classes.keys():
                if classes[key] > max_value:
                    max_class = key
                    max_value = classes[max_class]
                
            preds.append(max_class)

        return preds