;; Implementation based on https://github.com/fbeilstein/simplest_smo_ever

(ns models.svm
  (:require
   [clojure.core.matrix :as m]
   [utils.ovr :refer [ovr]]))


(defn linear [x y]
  (->> y
       (m/transpose)
       (m/dot x)))

(defn -poly [x y degree] (m/pow (linear x y) degree))

(defn poly [degree] #(-poly %1 %2 degree))

;; FIXME boken
(defn -rbf [x y gamma]
  (let [diff (m/sub y (m/reshape x [1 (m/dimension-count x 0)]))]
    (-> diff
        (m/pow 2)
        (m/esum)
        (m/mul (- gamma))
        #_{:clj-kondo/ignore [:unresolved-var]}
        (m/exp))))

(defn rbf [gamma] #(-rbf %1 %2 gamma))


(defn predict
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
        K (m/mul
           (kernel X X)
           (m/transpose Y)
           Y)]
    (loop [iter 0
           idxM 0
           lambdas (m/zero-array [col_num])]
      (cond
        (< n_iter iter)
        (let [idx (reduce
                   #(if (and (> (m/mget lambdas %2) 1e-15)
                             (not (zero? (m/mget lambdas %2))))
                      (conj %1 %2) %1)
                   []
                   (range (first (m/shape lambdas))))
              filter (fn [v]
                       (m/matrix
                        (reduce #(conj %1 (m/mget v %2))
                                [] idx)))
              filtered_K (filter K)
              filtered_Y (filter Y)
              temp_var (-> filtered_K
                           (m/mmul lambdas)
                           (m/esum)
                           (m/add -1)
                           (m/mul filtered_Y))
              mean (/ (m/esum temp_var) (first (m/shape temp_var)))]
          #(predict kernel X Y lambdas mean %))
        (< idxM col_num)
        (do
          (m/set-current-implementation :persistent-vector)
          (let [idxL (rand-int col_num)
                Q (m/matrix
                   [[(m/mget K idxM idxM) (m/mget K idxM idxL)]
                    [(m/mget K idxL idxM) (m/mget K idxL idxL)]])
                v0 (m/matrix [(m/mget lambdas idxM) (m/mget lambdas idxL)])
                k0 (- 1 (m/esum (m/mul lambdas [(m/mget K idxM) (m/mget K idxL)])))
                u (m/matrix [(- (m/mget Y idxL)) (m/mget Y idxM)])
                t_max (m/div (m/dot k0 u) (+ (m/dot (m/dot Q u) u) 1e-15))
                [new_idxM new_idxL] (->> (-restrict-to-square C t_max v0 u)
                                         (m/mul u)
                                         (m/add v0))]
            (recur iter (inc idxM)
                   (reduce #(m/mset %1 (first %2) (second %2))
                           lambdas
                           [[idxM new_idxM] [idxL new_idxL]]))))
        :else (recur (inc iter) 0 lambdas)))))


(defn core [kernel C n_iter]
  #(-fit kernel C n_iter %1 %2))

;; FIXME boken
(def svm
  (comp ovr core))