## ML/AIOps (Deployment, Monitoring, Evaluation)
Key components:

a. CI/CD (Continuous Integration/Continuous Deployment)
  * CI: Automatically tests new code changes (e.g., data preprocessing scripts, model training code) to ensure they don't break the system.
  * CD: Automates the deployment of a new model or a new version of the serving code to a staging or production environment.

b. Versioning and Registry
  * Model Registry: A centralized repository to store and manage different versions of trained models. This allows you to track which model version is in production and easily roll back if necessary.
  * Data and Code Versioning: Just as code needs to be versioned, so does data. Tools like DVC (Data Version Control) help track changes in datasets.

c. Monitoring
  * Model Performance Monitoring: Once a model is in production, it's crucial to monitor its performance. This includes:
  * Drift Detection: Monitoring for data drift (when the distribution of new data changes) or concept drift (when the relationship between input variables and the target variable changes).
  * Prediction Quality: Tracking the model's accuracy, precision, and other metrics in the production environment.
  * Infrastructure Monitoring: Monitoring the health of the underlying infrastructure, such as CPU and memory usage of the serving endpoints.

d. Automated Retraining
  * Trigger: When monitoring detects model performance degradation (e.g., a drop in accuracy or significant data drift), the MLOps pipeline can be automatically triggered to retrain the model.
  * Process: The pipeline uses a fresh dataset to retrain the model, evaluates the new version, and then automatically deploys it if it meets the performance criteria.



| Tool/Project | Links                            | Description                                       |
| ------------ | -------------------------------- | ------------------------------------------------- |
| langfuse     | [doc](https://langfuse.com/docs) | Observability, Pompot Optimization, and Evalution |
