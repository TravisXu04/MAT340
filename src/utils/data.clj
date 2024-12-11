(ns utils.data
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]))


(defn read-csv
  "Reads a CSV file and returns its contents as a sequence of rows.
   
   Each row is represented as a vector of strings."
  [filename]
  (with-open [reader (io/reader filename)]
    (doall
     (csv/read-csv reader))))


(defn split-X-Y
  "Splits the raw data into features (X) and labels (Y).
   - `raw`: Sequence of rows read from a CSV, where each row is a vector of strings.
   - `header`: Boolean indicating if the first row is a header (should be skipped).
   - `get-Y` (optional): The function uses to get the Y column (default: `#(last %)`).
    
   Returns a vector `[X Y]` where:
   - `X` is a vector of vectors representing the feature columns.
   - `Y` is a vector containing the labels."
  ([raw header get-Y]
   (let [data (if header (drop 1 raw) raw)
         X (mapv #(mapv (fn [x] (parse-double x)) (butlast %)) data)
         Y (reduce #(conj %1 (get-Y %2)) [] data)]
     [X Y]))
  ([raw header] (split-X-Y raw header #(last %))))


(defn split-train-test
  "Splits the dataset into training and testing sets.
   - `X`: Vector of feature vectors.
   - `Y`: Vector of labels.
   - `size`: Proportion of data to use for training (a number between 0 and 1).
   
   Returns a vector `[X_train Y_train X_test Y_test]` where:
   - `X_train` and `Y_train` are the training feature and label vectors.
   - `X_test` and `Y_test` are the testing feature and label vectors."
  [X Y size]
  (let [grouped (group-by #(% 1) (map-indexed vector Y))
        split-group
        (fn [[_ indices]]
          (let [total (count indices)
                train-count (int (* total size))
                train-idx (mapv #(% 0) (take train-count indices))
                test-idx (mapv  #(% 0) (drop train-count indices))]
            [train-idx test-idx]))
        splits (map split-group grouped)
        train-indices (mapcat #(% 0) splits)
        test-indices (mapcat #(% 1) splits)
        X_train (mapv X train-indices)
        Y_train (mapv Y train-indices)
        X_test (mapv X test-indices)
        Y_test (mapv Y test-indices)]
    [X_train Y_train X_test Y_test]))