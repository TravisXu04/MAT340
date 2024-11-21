(ns models.perceptron
  (:require
   [clojure.core.matrix :as m]
   [utils.ovr :refer [ovr]]))

(def -unshift-1 #(vec (cons 1 %)))

(defn -net-input-one [W x]
  (m/dot (-unshift-1 x) W))

(defn -predict-one [W x]
  (if (> (-net-input-one W x) 0) 1 -1))

(defn -net-input [W X]
  (->> W
       (drop 1)
       (m/dot X)
       (m/add (first W))))

(defn -predict [W X]
  (mapv #(if (> % 0) 1 -1) (-net-input W X)))


(defn -update-weight [eta W x y]
  (let [update  (->> x
                     (-predict-one W)
                     (- y)
                     (* eta)) ; calculate the update value
        x_c (-unshift-1 x)] ; new vector with 1 added to the front of the x
    (->> x_c
         (m/mul update)
         (m/add W))))


(defn -fit [eta n_iter X Y]
  (let [[sp_num col_num] (m/shape X)]
    (loop [iter 0 W  (-> col_num
                         (inc)
                         (repeat 0)
                         (vec))]
      (if (< n_iter iter) ; loop for n_iter times
        #(-predict W %1)
        (recur
         (inc iter) ; increment the iter counter
         (reduce ; loop through all the traning data and update the weights
          (fn [W idx]
            (-update-weight eta W
                            (X idx) (Y idx)))
          W (range sp_num)))))))


(defn core
  "Creates a perceptron training function with specified parameters.
   - `eta`: The learning rate.
   - `n_iter`: The number of training epochs.

   Returns:
   - The fitting function takes two arguments: `X` (training data) and `Y` (labels).
   - `Y` can only contain values `1` or `-1`; multiclass support is not available in this function.
   - It performs training using the provided `eta` and `n_iter` values, fitting the model to `X` and `Y`."
  [eta n_iter]
  #(-fit eta n_iter %1 %2))



(def perceptron
  "A basic Perceptron implementation.
  
   Parameters:
   - `eta`: The learning rate.
   - `n_iter`: The number of training epochs.

   Returns:
   - The fitting function takes the training data `X` and labels `Y`, and produces a prediction function.
   - The prediction function accepts input `X` and returns the predicted labels `Y`.

   This model is wrapped in a One-vs-Rest (OVR) strategy, enabling multi-class classification and compatibility with any set of target values `Y`. For a standalone Perceptron without OVR, use the `model.perceptron/core` function directly."
  (comp ovr core))
