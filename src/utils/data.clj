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


