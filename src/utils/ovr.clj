(ns utils.ovr)


(defn -pair-filter [cfn target pair]
  (reduce
   #(if (cfn (target %2))
      [(conj (first %1) (target %2))
       (conj (last %1) (pair %2))] %1)
   [[] []] (range (count target))))

(defn -sequential-replace [orig new]
  (loop [iter 0 idx 0 res []]
    (if (< iter (count orig))
      (if (= -1 (orig iter))
        (recur (inc iter) (inc idx) (conj res (get new idx)))
        (recur (inc iter) idx (conj res (get orig iter))))
      res)))



(defn ovr
  "Wraps a binary classification model to support multiclass classification using a One-vs-Rest (OVR) strategy.

   - `model`: A binary classification model that classifies inputs as either `1` or `-1`.

   - Returns a fit function that takes:
     - `X`: Training data.
     - `Y`: Labels.
     - The returned function produces predictions for new data `X`, assigning the appropriate class labels based on the trained models."
  [model]
  (fn [X Y] ; fit function
    (let [classes (vec (reduce #(conj %1 %2) #{} Y))
          models (loop [res [] set #{}] ; train each class one by one
                   (if (< (inc (count res)) (count classes))
                     (let [class (classes (count res))
                           [Y_filtered X_filtered] (-pair-filter #(not (contains? set %)) Y X) ; filter the classes that already trained
                           encoded (mapv #(if (= % class) 1 -1) Y_filtered) ; encoded the labels into 1 and -1
                           trained (model X_filtered encoded)] ; train the model
                       (recur (conj res [class trained]) (conj set class))) ; add to the vector of trained models
                     res))]
      (fn [X] ; predict function
        (mapv #(if (= % -1) (last classes) %) ; the remaining -1 is the last class which don't have its own model 
              (reduce (fn [prev model_pair] ; predict each class one by one
                        (let [[class model] model_pair]
                          (-sequential-replace prev ; insert into the final result
                                               (mapv #(if (= % -1) % class) ; replace all the 1 with the class name
                                                     (model (last (-pair-filter #(= -1 %) prev X))))))) ; predict here
                      (vec (repeat (count X) -1)) models))))))