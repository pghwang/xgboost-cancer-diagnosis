# Machine Learning Cancer Diagnosis (eXtreme Gradient Boosting with Cross-Validation)
This project creates an eXtreme Gradient Boosting (XGBoost) machine learning model to determine which genes best predict whether a patient will develop two cancer symptoms: 

    1. Perineural Invasion (PNI)      - Invasion of cancer to the space surrounding a nerve
    2. Lymphovascular Invasion (LVI)  - Invasion of cancer to lymphatics or blood vessels

The user will input two datasets 
(row: patient, col: normalized TPM counts per gene, 1st col: whether patient has PNI/LVI or not):

    1. 80% Training Dataset           - To train the XGBoost model and determine the optimal parameters
    2. 20% Testing Dataset            - To test the XGBoost model and evaluate the ROC AUC (Reciever Operating Characteristic Area Under Curve) metric

This project will output four files:

    1. Probability Matrix (Training)  - According to XGBoost, % likelihood that a given patient (in the training dataset) has PNI/LVI
    2. Probability Matrix (Testing)   - According to XGBoost, % likelihood that a given patient (in the testing dataset) has PNI/LVI
    3. Feature Importance             - Ranking of genes, according to its ability to predict PNI/LVI
    4. ROC Curve                      - X-axis: False positive rate, Y-axis: True positive rate

## Best Practices

1. Accuracy significantly improves when DESeq2 differential gene expression analysis is performed on the 80% training dataset before running this project. After running DESeq2, user should remove all genes whose P-adjusted values are lower than a threshold (e.g. 0.001, 0.005, 0.1, 0.5). This will ensure that only optimal genes are used for the XGBoost model. An example DESeq2 R script can be found in my GitHub.
2. Use tune_depth_weight(), then tune_gamma(), then tune_subsample_colsample() to determine the optimal parameters for the XGBoost model. Make sure to update parameters in the XGBoost() function after running all three tuning functions.
3. Take advantage of early_stopping_rounds, to ensure that the cross-validation model does not overfit to the evaluation dataset (20% of the training dataset).

## Results

    PNI:
        Training AUC:       1.000000
        Evaluation AUC:     0.782007
        Testing AUC:        0.770139
        Training Precision: 0.983871
        Testing Precision:  0.746193
        Training Recall:    1.000000
        Testing Recall:     0.849711
    LVI:
        Training AUC:       1.000000
        Evaluation AUC:     0.711158
        Testing AUC:        0.626445
        Training Precision: 1.000000
        Testing Precision:  0.532710
        Training Recall:    0.998384
        Testing Recall:     0.431818

## Best Parameters (Using GridSearchCV)

    PNI:
        xgb1 = XGBClassifier(
            learning_rate = 0.01,
            n_estimators = 5000,
            max_depth = 7,
            min_child_weight = 3,
            gamma = 0.2,
            subsample = 0.8,
            colsample_bytree = 0.9,
            objective = 'binary:logistic',
            nthread = 4,
            scale_pos_weight = 1,
            seed = 27)
    LVI:
        xgb1 = XGBClassifier(
            learning_rate = 0.01,
            n_estimators = 5000,
            max_depth = 9,
            min_child_weight = 1,
            gamma = 0.3,
            subsample = 0.5,
            colsample_bytree = 0.6,
            objective = 'binary:logistic',
            nthread = 4,
            scale_pos_weight = 1,
            seed = 27)

## Author

* **Peter Hwang** - [pghwang](https://github.com/pghwang)

## Acknowledgments

* This project has been created for an undergraduate research project under the Broad Institute of MIT and Harvard.
* Special thanks to Jimmy Guo and Hannah Hoffman for the guidance and support!

## References

XGBoost CV: https://www.kaggle.com/cast42/xg-cv
XGBoost Parameter Tuning: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
