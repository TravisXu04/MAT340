;; Implementation based on https://github.com/fbeilstein/simplest_smo_ever which is based on https://cs229.stanford.edu/materials/smo.pdf

(ns models.svm
  (:require
   [clojure.core.matrix :as m]
   [utils.ovr :refer [ovr]]))


(defn linear 
  "Defines a linear kernel for SVM.
  
   Parameters:
   - `x`: The first input matrix or vector.
   - `y`: The second input matrix or vector.
  
   Returns:
   - The dot product of `x` and the transpose of `y`.
  
   Usage:
   - Use this kernel in the `core` or `svm` functions for linear separability in SVM models."
  [x y]
  (->> y
       (m/transpose)
       (m/dot x)))

(defn -poly [x y degree] 
  (m/pow (linear x y) degree))

(defn poly
  "Creates a polynomial kernel for SVM.
  
   Parameters:
   - `degree`: The degree of the polynomial.
  
   Returns:
   - A kernel function that takes two arguments `x` and `y` and computes the polynomial transformation of their dot product.
  
   Usage:
   - Use this kernel in the `core` or `svm` functions for non-linear separability with polynomial features."
  [degree]
  #(-poly %1 %2 degree))

(defn -new-axis [X] 
  (m/matrix 
   (mapv #(vec [%])
         (m/to-nested-vectors X))))

(defn -special-sum [m]
  (m/matrix
   (mapv
    #(mapv m/esum %)
    m)))

(defn -special-sub [x y]
  (m/matrix
   (mapv
    #(mapv
      (fn [i_y]
        (m/sub (first %) i_y))
      x)
    y)))

(defn -rbf [x y gamma]
  (let [diff (-special-sub y (-new-axis x))]
    (-> diff
        (m/pow 2)
        (-special-sum)
        (m/mul (- gamma))
        #_{:clj-kondo/ignore [:unresolved-var]}
        (m/exp))))

(defn rbf 
  "Creates an RBF (Radial Basis Function) kernel for SVM.
  
   Parameters:
   - `gamma`: The gamma parameter for the RBF kernel. Controls the influence of a single training example.
  
   Returns:
   - A kernel function that takes two arguments `x` and `y` and computes the RBF transformation.
  
   Usage:
   - Use this kernel in the `core` or `svm` functions for non-linear separability with RBF kernels."
  [gamma] 
  #(-rbf %1 %2 gamma))

(defn -predict
  [kernel train_X train_Y train_lambdas train_b X]
  (if (= (first (m/shape X)) 0)
    []
    (let [-row-sum (fn [m]
                     (m/matrix
                      (mapv #(m/esum (m/mget m %))
                            (range (first (m/shape m))))))
          -add-b (fn [m] (m/emap #(+ train_b %) m))
          decision
          (-> X
              (kernel train_X)
              (m/mul train_Y train_lambdas)
              (-row-sum)
              (-add-b))]
      (m/to-vector
       (m/emap
        #(if (> % 0) 1 -1)
        decision)))))


(defn -restrict-to-square [C t v0 u]
  (let [get-t (fn [i_t idx]
                (-> v0
                    (m/add (m/mul i_t u))
                    (m/clamp 0 C)
                    (m/sub v0)
                    (m/mget idx)
                    (m/div (m/mget u idx))))
        new_t (get-t t 1)]
    (get-t new_t 0)))


(defn -fit [kernel C n_iter X Y] 
  (let [Y (-> Y
              (m/mul 2)
              (m/sub 1))
        [col_num _] (m/shape Y)
        new_axis_y (-new-axis Y)
        K (m/mmul
           (kernel X X)
           new_axis_y
           (m/transpose new_axis_y))]
    (loop [iter 0
           idxM 0
           lambdas (m/zero-array [col_num])]
      (m/set-current-implementation :persistent-vector)
      (cond
        (< n_iter iter)
        (let [idx (reduce
                   #(if (and (> (m/mget lambdas %2) 1e-15)
                             (not (zero? (m/mget lambdas %2))))
                      (conj %1 %2) %1)
                   []
                   (range (first (m/shape lambdas))))
              filtered_K (m/matrix
                          (reduce #(conj %1 (m/get-column K %2))
                                  [] idx))
              filtered_Y (m/matrix
                          (reduce #(conj %1 (get Y %2))
                                  [] idx))
              temp_var (-> filtered_K
                           (m/mmul lambdas)
                           (m/esum)
                           (+ -1)
                           (m/mul filtered_Y))
              mean (/ (m/esum temp_var) (first (m/shape temp_var)))]
          #(-predict kernel X Y lambdas mean %))
        (< idxM col_num)
        (let [idxL
              (loop [result (rand-int col_num)]
                (if (= result idxM)
                  (recur (rand-int col_num))
                  result))
              Q (m/matrix
                 [[(m/mget K idxM idxM) (m/mget K idxM idxL)]
                  [(m/mget K idxL idxM) (m/mget K idxL idxL)]])
              v0 (m/matrix
                  [(m/mget lambdas idxM) (m/mget lambdas idxL)])
              k0 (->> [(m/get-row K idxM) (m/get-row K idxL)]
                      (m/mul lambdas)
                      (mapv m/esum)
                      (m/sub 1))
              u (m/matrix [(m/sub (m/mget Y idxL)) (m/mget Y idxM)])
              t_max (/ (m/dot k0 u)
                       (+ (m/dot (m/dot Q u) u) 1e-15))
              [new_idxM new_idxL] (->> (-restrict-to-square C t_max v0 u)
                                       (m/mul u)
                                       (m/add v0))]
          (recur iter (inc idxM)
                 (reduce #(m/mset %1 (first %2) (second %2))
                         lambdas
                         [[idxM new_idxM] [idxL new_idxL]])))
        :else (recur (inc iter) 0 lambdas)))))


(defn core 
    "Creates an SVM (Support Vector Machine) training function with specified parameters.
     
     Parameters:
     - `kernel`: The kernel function to use for the SVM. Common options include:
       - `linear`: A linear kernel for linear separability.
       - `poly`: A polynomial kernel for non-linear separability (see `poly`).
       - `rbf`: A radial basis function (Gaussian) kernel for highly non-linear separability (see `rbf`).
     - `C`: The regularization parameter. Higher values increase the penalty for misclassified points.
     - `n_iter`: The number of training epochs.
  
     Returns:
     - The fitting function takes the training data `X` and labels `Y`, and produces a prediction function.
     - The prediction function accepts input `X` and returns the predicted labels `Y`."
  [kernel C n_iter]
  #(-fit kernel C n_iter %1 %2))

;; FIXME low accuracy
(def svm
  "An SVM (Support Vector Machine) implementation wrapped with One-vs-Rest (OVR) strategy.
  
   Parameters:
   - `kernel`: The kernel function to use for the SVM (see `core` for options).
   - `C`: The regularization parameter.
   - `n_iter`: The number of training epochs.
  
   Returns:
   - The fitting function takes the training data `X` and labels `Y` and produces a prediction function.
   - The prediction function accepts input `X` and returns the predicted labels `Y`.
   
   This model leverages the One-vs-Rest strategy, enabling multi-class classification. For binary classification or standalone SVM training, use the `core` function directly."
  (comp ovr core))