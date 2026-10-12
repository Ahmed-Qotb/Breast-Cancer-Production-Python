from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from production_app.logging_config import logger


def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"],
    )
    matrix = confusion_matrix(y_test, y_pred)

    logger.info("Model evaluation completed")
    logger.info("Accuracy: %.2f%%", accuracy * 100)
    logger.info("\n%s", report)
    logger.info("\nConfusion Matrix:\n%s", matrix)

    return accuracy, report, matrix
