from production_app.config import DATA_PATH, MODEL_PATH
from production_app.data import get_dataset, split_dataset
from production_app.evaluation import evaluate_model
from production_app.logging_config import logger
from production_app.model_service import save_model
from production_app.training import train_model


def main():
    # 1. Load the dataset
    df = get_dataset(DATA_PATH)

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(df)

    # 3. Train the model
    model, X_test, y_test = train_model(X_train, X_test, y_train, y_test)

    # 4. Evaluate the model
    accuracy, report, matrix = evaluate_model(model, X_test, y_test)

    # 5. Save the trained model
    save_model(model, MODEL_PATH)

    logger.info("Workflow completed successfully")
    print(f"Accuracy: {accuracy:.2%}")
    print(report)
    print(matrix)


if __name__ == "__main__":
    main()
