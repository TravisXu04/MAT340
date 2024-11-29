(ns preprocessing
  (:require
   [clojure.core.matrix.stats :as s]
   [clojure.core.matrix :as m]))

(def dummy-preprocessing
  "A no-op preprocessing function that does not modify the input data.
     
   Usage:
   - Takes any input data and returns it unchanged.
     
   Returns:
   - A function that, when called with data `x`, simply returns `x` without any transformations."
  (fn [_] (fn [x] x)))

(defn -min-max-scaler [x]
  (let [max (apply max x)
        min (apply min x)]
    (fn [val]
      (mapv
       #(double (/ (- % min) (- max min)))
       val))))

(defn -standard-scaler [x]
  (let [mean (s/mean x)
        sd (s/sd x)]
    (fn [val]
      (mapv
       #(double (/ (- % mean) sd))
       val))))

(defn v2m [v]
  (fn [X]
    (let [scalers (mapv #(v %) (m/transpose X))]
      (fn [val]
        (let [X_t (m/transpose val)]
          (m/transpose
           (mapv #((scalers %) (X_t %))
                 (range (count X_t)))))))))

(def min-max-scaler
  "Creates a Min-Max Scaler function for scaling data to a [0, 1] range for each feature.

   Usage:
   - Takes a dataset `X` and returns a scaling function to normalize individual data points.
   
   Returns:
   - A scaling function that, when given a new data point `val`, scales each feature of `val` 
     based on the min and max values of each feature in the original dataset `X`."
  (v2m -min-max-scaler))


(def standard-scaler
  "Creates a Standard Scaler function for standardizing data to have a mean of 0 and 
   standard deviation of 1 for each feature.
   
   Usage:
   - Takes a dataset `X` and returns a scaling function to standardize individual data points.
   
   Returns:
   - A scaling function that, when given a new data point `val`, standardizes each feature of `val` 
     based on the mean and standard deviation of each feature in the original dataset `X`."
  (v2m -standard-scaler))

