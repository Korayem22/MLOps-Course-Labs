
import mlflow
import pickle as pkl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
def log(uri,expermint,col_transf,y_pred,y_test,run_name,model,prams,sign):
    print(f"logging {run_name}") 
    ### Set the tracking URI for MLflow
    print(uri)
    mlflow.set_tracking_uri(uri)
    ### Set the experiment name
    mlflow.set_experiment(expermint)
    mlflow.start_run(run_name=run_name)
    ### Train the model
    pkl.dump(col_transf, open("transformer.pkl", "wb"))
    mlflow.log_artifact("transformer.pkl")
    mlflow.sklearn.log_model(model,run_name,signature=sign)
    mlflow.log_params(prams)
    ### Log metrics after calculating them
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))
    ### Log tag
    mlflow.set_tag("model", run_name)
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    conf_mat_disp.figure_
    plt.savefig("confusion_matrix.png")
    # Log the image as an artifact in MLflow
    mlflow.log_artifact("confusion_matrix.png")
    plt.show()
    mlflow.end_run()