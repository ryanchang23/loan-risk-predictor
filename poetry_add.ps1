# poetry add absl-py==1.4.0 astor==0.8.1 cycler==0.11.0 decision-tree==0.4 \
#     DecisionTree==3.2.1 entropy-estimators==0.0.1 fonttools==4.38.0 gast==0.5.4 \
#     grpcio==1.56.0 h5py==3.8.0 importlib-metadata==6.7.0 joblib==1.3.1 \
#     Keras==2.3.1 Keras-Applications==1.0.8 Keras-Preprocessing==1.1.2 \
#     kiwisolver==1.4.5 KNN==1.0.0 legacy==0.1.6 lightgbm==4.0.0 Markdown==3.4.3 \
#     MarkupSafe==2.1.3 matplotlib==3.5.3 mock==5.0.2 numpy==1.21.6 packaging==23.1 \
#     pandas==0.25.3 parameters==0.2.1 Pillow==9.5.0 protobuf==3.6.0 \
#     pyparsing==3.1.1 PySimpleGUI==4.60.5 python-dateutil==2.8.2 pytz==2023.3 \
#     PyYAML==6.0 scikit-learn==1.0.2 scipy==1.5.1 six==1.16.0 \
#     sklearn==0.0.post5 tensorboard==1.13.1 tensorflow==1.13.2 \
#     tensorflow-estimator==1.13.0 termcolor==2.3.0 threadpoolctl==3.1.0 \
#     typing_extensions==4.7.1 Werkzeug==2.2.3 zipp==3.15.0


# poetry remove tensorflow tensorflow-estimator tensorboard
# poetry add tensorflow==2.15.0
# poetry add scikit-learn@latest
# poetry add pandas@latest
# poetry add protobuf@latest
# poetry add matplotlib@latest
# poetry install

# poetry add absl-py astor cycler decision-tree DecisionTree entropy-estimators fonttools gast grpcio h5py importlib-metadata joblib Keras Keras-Applications Keras-Preprocessing kiwisolver KNN legacy lightgbm Markdown MarkupSafe matplotlib mock numpy packaging pandas parameters Pillow protobuf pyparsing PySimpleGUI python-dateutil pytz PyYAML scikit-learn scipy six sklearn tensorboard tensorflow tensorflow-estimator termcolor threadpoolctl typing_extensions Werkzeug zipp
# poetry remove numpy tensorflow

poetry add absl-py astor cycler decision-tree DecisionTree entropy-estimators fonttools gast grpcio h5py importlib-metadata joblib Keras Keras-Applications Keras-Preprocessing kiwisolver KNN legacy lightgbm Markdown MarkupSafe matplotlib mock numpy packaging pandas parameters Pillow protobuf pyparsing PySimpleGUI python-dateutil pytz PyYAML scikit-learn scipy six tensorboard tensorflow tensorflow-estimator termcolor threadpoolctl typing_extensions Werkzeug zipp
# poetry add absl-py astor cycler decision-tree DecisionTree entropy-estimators fonttools gast grpcio h5py importlib-metadata joblib Keras Keras-Applications Keras-Preprocessing kiwisolver KNN legacy lightgbm Markdown MarkupSafe matplotlib mock numpy packaging pandas parameters Pillow protobuf pyparsing PySimpleGUI python-dateutil pytz PyYAML scikit-learn scipy six tensorboard tensorflow tensorflow-estimator termcolor threadpoolctl typing_extensions Werkzeug zipp
poetry add tensorflow@2.18.0
poetry add numpy@1.26.4
poetry add PySimpleGUI
poetry add scikit-learn 
poetry add matplotlib

poetry add absl-py astor cycler decision-tree DecisionTree entropy-estimators fonttools gast grpcio h5py importlib-metadata joblib Keras Keras-Applications Keras-Preprocessing kiwisolver KNN legacy lightgbm Markdown MarkupSafe matplotlib mock numpy packaging pandas parameters Pillow protobuf pyparsing PySimpleGUI python-dateutil pytz PyYAML scipy six tensorboard tensorflow tensorflow-estimator termcolor threadpoolctl typing_extensions Werkzeug zipp

poetry add pandas 
poetry add tensorboard tensorflow tensorflow-estimator termcolor threadpoolctl typing_extensions Werkzeug zipp
poetry add lightgbm
poetry add legacy
poetry add xgboost 



# other
poetry add tqdm
poetry add customtkinter 



# reinstall 
poetry env remove python 
poetry cache claer . --all
poetry install --no-root


# (if have lock issue)
poetry env remove python 
poetry cache clear . --all
poetry lock --no-update  
poetry install  --no-root



