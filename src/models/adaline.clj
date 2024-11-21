(ns models.adaline
  (:require
   [clojure.core.matrix :as m]
   [models.perceptron :refer [-net-input -predict]]
   [utils.ovr :refer [ovr]]))



(defn -update-weight [eta W X Y]
  (let [output (-net-input W X) ; get the net output
        errors  (m/sub Y output) ; calculate the error
        update (-> X
                   (m/transpose)
                   (m/dot errors)
                   (m/mul eta))] ; calculate the update values
    ;; (println (/ (reduce + (mapv #(* % %) errors)) 2)) ; FOR DEVELOPMENT print the cost
    (->>  update
          (m/add (drop 1 W))
          (cons
           (->> errors
                (m/esum)
                (* eta)
                (+ (W 0))))
          (vec))))


(defn -fit [eta n_iter X Y]
  (let [[_ col_num] (m/shape X)]
    (loop [iter 0 W (-> col_num
                        (inc)
                        (repeat 0)
                        (vec))]
      (if (< n_iter iter) ; loop for n_iter times
        #(-predict W %1)
        (recur
         (inc iter) ; increment the iter counter
         (-update-weight eta W X Y)))))) ; update the weights


(defn core
  "Creates a adaline training function with specified parameters.
   - `eta`: The learning rate.
   - `n_iter`: The number of training epochs.

   Returns:
   - The fitting function takes two arguments: `X` (training data) and `Y` (labels).
   - `Y` can only contain values `1` or `-1`; multiclass support is not available in this function.
   - It performs training using the provided `eta` and `n_iter` values, fitting the model to `X` and `Y`."
  [eta n_iter]
  #(-fit eta n_iter %1 %2))



(def adaline
  "A basic Adaline implementation.
  
   Parameters:
   - `eta`: The learning rate.
   - `n_iter`: The number of training epochs.

   Returns:
   - The fitting function takes the training data `X` and labels `Y`, and produces a prediction function.
   - The prediction function accepts input `X` and returns the predicted labels `Y`.

   This model is wrapped in a One-vs-Rest (OVR) strategy, enabling multi-class classification and compatibility with any set of target values `Y`. For a standalone Adaline without OVR, use the `model.adaline/core` function directly."
  (comp ovr core))
