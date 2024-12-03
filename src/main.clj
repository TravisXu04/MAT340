(ns main
  (:require
   [models.adaline :refer [adaline]]
   [models.knn :refer [knn]]
   [models.logistic-regression :refer [logistic-regression]]
   [models.perceptron :refer [perceptron]]
   [models.svm :refer [linear poly rbf svm]]
   [utils.test :refer [test-model]]))


#_{:clj-kondo/ignore [:clojure-lsp/unused-public-var]}
(defn main [_]
  (let [models
        {:preceptron (perceptron 0.001 50)
         :adaline (adaline 0.001 50)
         :logistic-regression (logistic-regression 0.001 10 0.1)
         :knn (knn 5 2)
         :svm-linear (svm linear 10 50)
         :svm-poly (svm (poly 3) 5 50)
         :svm-rbf (svm (rbf 1) 10 50)}]
    (mapv
     (fn [[k v]]
       (println "Model" k)
       (test-model v)
       (println))
     models)))
 

