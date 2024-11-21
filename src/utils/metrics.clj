(ns utils.metrics
  (:require
   [clojure.pprint :refer [print-table]]
   [clojure.core.matrix :as m]))

(defn accuracy
  "Calculates the accuracy of predictions compared to the actual labels.
   - `Y_true`: Vector of actual labels.
   - `Y_pred`: Vector of predicted labels.
    
   Returns a double representing the proportion of correct predictions."
  [Y_true Y_pred]
  ((comp double /)
   (reduce
    #(if (= (get Y_pred %2) (get Y_true %2)) (inc %1) %1)
    (range (inc (count Y_true))))
   (count Y_true)))

(defn confusion-matrix
  "Computes the confusion matrix for a set of true labels and predicted labels.
   
   Parameters:
   - `Y_true`: A sequence of true class labels.
   - `Y_pred`: A sequence of predicted class labels.
     
   Returns:
   - A matrix (vector of vectors) representing the confusion matrix. 
   Each row corresponds to an actual class, and each column corresponds to a predicted class. 
   The value at position `(i, j)` indicates how many times class `i` was predicted as class `j`."
  [Y_true Y_pred]
  (let [classes (sort (set Y_true))
        class-index (zipmap classes (range))
        matrix (vec (repeat (count classes) (vec (repeat (count classes) 0))))]
    [classes
     (reduce
      (fn [mat [true-label pred-label]]
        (let [i (class-index true-label)
              j (class-index pred-label)]
          (update-in mat [i j] inc)))
      matrix
      (map vector Y_true Y_pred))]))

(defn precision
  "Calculates the precision for each class in a multi-class confusion matrix.
   
   Precision is defined as: 
   Precision = True Positive / (True Positive + False Positive)
  
   Parameters:
   - `classes`: A collection of class labels.
   - `confusion-matrix`: A 2D matrix where each element [i][j] represents the number of instances that were predicted as class j but actually belong to class i.
     
   Returns:
   - A map where each key is a class label, and the value is the precision for that class."
  [classes confusion-matrix]
  (let [n-classes (count classes)
        get-tp (fn [class] (get-in confusion-matrix [class class]))
        get-fp (fn [class]
                 (reduce + (map #(get-in confusion-matrix [% class] 0)
                                (remove #(= % class) (range n-classes)))))]
    (reduce
     (fn [result class]
       (let [tp (get-tp class)
             fp (get-fp class)
             precision (if (> (+ tp fp) 0)
                         (/ tp (+ tp fp))
                         0)]
         (assoc result (nth classes class) (double precision))))
     {}
     (range n-classes))))


(defn recall
  "Calculates the recall for each class in a multi-class confusion matrix.
   
   Recall is defined as:
   Recall = True Positive / (True Positive + False Negative)
  
   Parameters:
   - `classes`: A collection of class labels.
   - `confusion-matrix`: A 2D matrix where each element [i][j] represents the number of instances that were predicted as class j but actually belong to class i.
 
   Returns:
   - A map where each key is a class label, and the value is the recall for that class."
  [classes confusion-matrix]
  (let [n-classes (count classes)
        get-tp (fn [class] (get-in confusion-matrix [class class]))
        get-fn (fn [class]
                 (reduce + (map #(get-in confusion-matrix [class %] 0)
                                (remove #(= % class) (range n-classes)))))]
    (reduce
     (fn [result class]
       (let [tp (get-tp class)
             fn (get-fn class)
             recall (if (> (+ tp fn) 0)
                      (/ tp (+ tp fn))
                      0)]
         (assoc result (nth classes class) (double recall))))
     {}
     (range n-classes))))

(defn f1-score
  "Calculates the F1-score for each class in a multi-class confusion matrix.
     
   F1-score is defined as:
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
 
   Parameters:
   - `classes`: A collection of class labels.
   - `confusion-matrix`: A 2D matrix where each element [i][j] represents the number of instances that were predicted as class j but actually belong to class i.
 
   Returns:
   - A map where each key is a class label, and the value is the F1-score for that class."
  [classes confusion-matrix]
  (let [precisions (precision classes confusion-matrix)
        recalls (recall classes confusion-matrix)]
    (reduce
     (fn [result class]
       (let [precision (get precisions class)
             recall (get recalls class)
             f1 (if (> (+ precision recall) 0)
                  (/ (* 2 precision recall) (+ precision recall))
                  0)]
         (assoc result class (double f1))))
     {}
     classes)))


(defn -round-to [n places]
  (/ (Math/round (* n (Math/pow 10 places)))
     (Math/pow 10 places)))

(defn classification-report
  "Generates and prints a classification report, including accuracy, confusion matrix, precision, recall, and F1-score.
   
   Parameters:
   - `Y_true`: The true labels for the data.
   - `Y_pred`: The predicted labels from the model.
   - `print_cm` (optional):  Boolean indicating if the confusion matrix should be printed (default: `false`)."
  ([Y_true Y_pred print_cm]
   (println "Accuracy:" (accuracy Y_true Y_pred))
   (let [[classes cm] (confusion-matrix Y_true Y_pred)
         precision (vals (precision classes cm))
         recall (vals (recall classes cm))
         f1_score (vals (f1-score classes cm))
         score_str ["Recall" "Precision" "F1-score"]
         score (m/transpose [precision recall f1_score])]
     #_{:clj-kondo/ignore [:missing-else-branch]}
     (if print_cm
       ((println "Confusion Matrix:")
        (print-table
         (mapv #(into {'* (nth classes %)}
                      (zipmap classes (nth cm %)))
               (range (count cm))))
        (println)))
     (println "Scores:")
     (print-table
      (mapv #(into {'* (nth classes %)}
                   (zipmap score_str
                           (mapv (fn [s] (-round-to s 2))
                                 (nth score %))))
            (range (count cm))))))
  ([Y_true Y_pred] (classification-report Y_true Y_pred false)))
