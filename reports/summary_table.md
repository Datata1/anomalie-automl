| Strategie | roc_auc | pr_auc | f1 | detection_delay | false_alarm_rate |
|---|---|---|---|---|---|
| default:ecod | 0.780 | 0.924 | 0.860 | 409.644 | 0.000 |
| default:iforest | 0.803 | 0.933 | 0.861 | 18.244 | 0.044 |
| default:ocsvm | 0.855 | 0.954 | 0.862 | 10.300 | 0.045 |
| default:pca | 0.854 | 0.954 | 0.863 | 14.578 | 0.042 |
| default:autoencoder | 0.846 | 0.952 | 0.860 | 5.256 | 0.098 |
| default:som | 0.850 | 0.953 | 0.860 | 5.044 | 0.110 |
| default:deep_svdd | 0.815 | 0.939 | 0.858 | 8.044 | 0.071 |
| hpo:iforest | 0.813 | 0.936 | 0.861 | 15.833 | 0.046 |
| ensemble:average | 0.845 | 0.950 | 0.862 | 33.011 | 0.015 |
| select_internal:pca | 0.854 | 0.954 | 0.863 | 14.578 | 0.042 |
| select_oracle:ocsvm | 0.855 | 0.954 | 0.862 | 10.300 | 0.045 |
