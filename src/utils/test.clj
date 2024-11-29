(ns utils.test
  (:require
   [decomposition.pca :refer [pca]]
   [utils.data :refer [read-csv split-X-Y]]
   [utils.metrics :refer [classification-report]]
   [preprocessing :refer [dummy-preprocessing standard-scaler]]))


(defn test-model
  "Tests a given model using optional dimensionality reduction and scaling.
   
   Parameters:
   - `model`: A function that takes training data `X` and labels `Y` and returns a trained model.
   - `dim_red` (optional): A dimensionality reduction function (e.g., PCA). Defaults to `pca` with 80% variance retention.
   - `scaler` (optional): A data scaler function (e.g., `standard-scaler`). Defaults to `standard-scaler`."
  ([model dim_red scaler]
   (let [[X Y] (split-X-Y (read-csv "datasets/iris.csv") true)
   ; Scaling
         trained_scaler (scaler X)
         X_std (trained_scaler X)
         ; Feature Extraction
         trained_dim_red (dim_red X_std)
         X_pca (trained_dim_red X_std)
         ; Training
         trained_model (model X_pca Y)]
     ; Prediction
     (classification-report
      Y (trained_model X_pca))))
  ([model] (test-model model (pca 0.8) standard-scaler)))

(defn test-model-no-scaler
  "Tests a given model without scaling the data.
  
  Parameters:
  - `model`: A function that takes training data `X` and labels `Y` and returns a trained model.
  - `dim_red` (optional): A dimensionality reduction function (e.g., PCA). Defaults to `pca` with 80% variance retention."
  ([model dem_red] (test-model model dem_red dummy-preprocessing))
  ([model] (test-model-no-scaler  model (pca 0.8))))

(defn test-model-no-dim-red
  "Tests a given model without applying dimensionality reduction.
  
  Parameters:
  - `model`: A function that takes training data `X` and labels `Y` and returns a trained model.
  - `scaler` (optional): A data scaler function (e.g., `standard-scaler`). Defaults to `standard-scaler`."
  ([model scaler] (test-model model dummy-preprocessing scaler))
  ([model] (test-model-no-dim-red  model  standard-scaler)))

(defn test-model-raw
  "Tests a given model without any data preprocessing.
  
  Parameters:
  - `model`: A function that takes training data `X` and labels `Y` and returns a trained model."
  [model]
  (test-model model dummy-preprocessing dummy-preprocessing))