(ns models.knn
  (:require
   [clojure.core.matrix :as m]))


(defn -euclidean-distance [vec1 vec2 p]
  (-> (m/sub vec1 vec2)
      (m/pow p)
      (m/esum)
      (abs)
      (Math/sqrt)))

(defn -get-nearest-k-neighbors [k p X_train Y_train x]
  (let [distances (map (fn [x_train y_train]
                         {:distance (-euclidean-distance x x_train p)
                          :label y_train})
                       X_train
                       Y_train)]
    (->> distances
         (sort-by :distance)
         (take k))))

(defn -predict-one [n_neighbors p X_train Y_train x]
  (let [neighbors (-get-nearest-k-neighbors n_neighbors p X_train Y_train x)
        votes (frequencies (map :label neighbors))]
    (->> votes
         (sort-by val >)
         (first)
         (key))))

(defn -predict [n_neighbors p X_train Y_train X]
  (mapv
   #(-predict-one n_neighbors p X_train Y_train %) X))


(defn knn
  "A k-Nearest Neighbors (k-NN) classifier implementation.
  
   Parameters:
   - `n_neighbors`: The number of neighbors to consider for making predictions.
   - `p`: The power parameter for the Minkowski distance metric.
  
   Returns:
   - A function that takes `X_train` (training data) and `Y_train` (training labels) as arguments, and produces a prediction function.
   - The prediction function accepts an input `X` and returns the predicted labels.
   
   This model supports multi-class classification and works with any set of target values `Y` by design."
  [n_neighbors p]
  (fn [X_train Y_train]
    #(-predict n_neighbors p X_train Y_train %)))