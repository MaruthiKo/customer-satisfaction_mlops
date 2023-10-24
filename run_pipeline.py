from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="./data/olist_customers_dataset.csv")

# mlflow ui --backend-store-ui "file:C:\Users\ASUS\AppData\Roaming\zenml\local_stores\1125313a-e0a4-41f2-a598-a7c2cf48cc03\mlruns"