(ns models.logistic-regression
  (:require
   [clojure.core.matrix :as m]
   [models.perceptron :refer [-net-input -predict]]
   [utils.ovr :refer [ovr]]))


(defn -activation [Z]
  (->> Z
       (m/sub)
       #_{:clj-kondo/ignore [:unresolved-var]}
       (m/exp)
       (m/add 1)
       (m/div 1)))


(defn -update-weight [eta C W X Y]
  (let [output (->> X
                    (-net-input W)
                    (-activation)) ; get the output
        errors  (m/sub Y output) ; calculate the error
        update
        (-> X (m/transpose)
            (m/dot errors)
            (m/sub
             (->> W
                  (drop 1)
                  (m/mul (/ 1 C))))
            (m/mul eta))] ; calculate the update values
    (->> update
         (m/add (drop 1 W))
         (cons
          (->> errors
               (m/esum)
               (* eta)
               (+ (W 0))))
         (vec))))


(defn -fit [eta n_iter C X Y]
  (let [[_ col_num] (m/shape X)]
    (loop [iter 0 W (-> col_num
                        (inc)
                        (repeat 0)
                        (vec))]
      (if (< n_iter iter) ; loop for n_iter times
        #(-predict W %1)
        (recur
         (inc iter) ; increment the iter counter
         (-update-weight eta C W X Y)))))) ; update the weights


(defn core
  "Creates a logistic regression training function with specified parameters.
   - `eta`: The learning rate.
   - `n_iter`: The number of training epochs.
   - `C`: The inverse of regularization strength; must be a positive float. 

   Returns:
   - The fitting function takes two arguments: `X` (training data) and `Y` (labels).
   - `Y` can only contain values `1` or `-1`; multiclass support is not available in this function.
   - It performs training using the provided `eta` and `n_iter` values, fitting the model to `X` and `Y`."
  [eta n_iter C]
  #(-fit eta n_iter C %1 %2))


;; TODO add option to return the pred_prob
(def logistic-regression
  "A basic Logistic Regression implementation.
  
   Parameters::
   - `eta`: The learning rate.
   - `n_iter`: The number of training epochs.
   - `C`: The inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization. 

   Returns:
   - The fitting function takes the training data `X` and labels `Y`, and produces a prediction function.
   - The prediction function accepts input `X` and returns the predicted labels `Y`.

   This model is wrapped in a One-vs-Rest (OVR) strategy, enabling multi-class classification and compatibility with any set of target values `Y`. For a standalone Adaline without OVR, use the `model.adaline/core` function directly."
  (comp ovr core))
