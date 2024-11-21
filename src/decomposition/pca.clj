(ns decomposition.pca
  (:require
   [clojure.core.matrix :as m ]
   [clojure.core.matrix.linear :refer [svd]]))



(defn pca
  "Performs Principal Component Analysis (PCA) for dimensionality reduction.
  
   Parameters:
   - `n_components`: Specifies the number of principal components to retain. 
     - If an integer >= 1, it directly sets the number of components.
     - If a float between 0 and 1, it represents the minimum proportion of variance to retain.

   Returns:
   - A function that, when given a data matrix `X`, computes the PCA of `X` 
     and returns a projection function for new data values."
  [n_components]
  (m/set-current-implementation :vectorz)
  (fn [X]
    (let [res (svd (m/matrix X)) s (:S res) Vt (:V* res)]
      (fn [val]
        (let [explained_variance (m/div s (- (first (m/shape X)) 1))
              explained_variance_ratio (m/div explained_variance (m/esum explained_variance))
              cal_n_components
              (if (>= n_components 1)
                n_components
                (loop [n 0]
                  (if (>= (m/esum (take n explained_variance_ratio)) n_components)
                    n
                    (recur (inc n)))))
              W (m/transpose (take cal_n_components Vt))]
          (m/dot val W))))))