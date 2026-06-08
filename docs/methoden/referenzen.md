# Referenzen

Zentrale, zitierfähige Quellen der Methodensammlung. Gruppiert nach Thema; Links zu
arXiv/PyPI/Repos, soweit vorhanden. Vor Übernahme in die PowerPoint Jahreszahlen/Autoren
final gegenprüfen.

## Datensatz (TEP)

- **Downs, J. J., & Vogel, E. F. (1993).** A plant-wide industrial process control problem.
  *Computers & Chemical Engineering, 17(3).* — Originaler TEP-Benchmark.
- **Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017).** Additional Tennessee
  Eastman Process Simulation Data for Anomaly Detection Evaluation. *Harvard Dataverse.* —
  Quelle der hier genutzten RData-Dateien.
- Kaggle-Spiegel: <https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset>

## AutoML — Grundlagen

- **Hutter, F., Kotthoff, L., & Vanschoren, J. (2019).** *Automated Machine Learning: Methods,
  Systems, Challenges.* Springer (Open Access). — Standardwerk: HPO, CASH, Meta-Learning.
- **Feurer, M. et al. (2015).** Efficient and Robust Automated Machine Learning (auto-sklearn).
  *NeurIPS.*
- **Feurer, M. et al. (2020/2022).** Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning.
  *JMLR.*
- **Li, L. et al. (2017).** Hyperband: A Novel Bandit-Based Approach to Hyperparameter
  Optimization. *JMLR.* — Multi-Fidelity.
- **Falkner, S., Klein, A., & Hutter, F. (2018).** BOHB: Robust and Efficient Hyperparameter
  Optimization at Scale. *ICML.*
- **Akiba, T. et al. (2019).** Optuna: A Next-generation Hyperparameter Optimization
  Framework. *KDD.* — <https://optuna.org>
- **Lindauer, M. et al. (2022).** SMAC3: A Versatile Bayesian Optimization Package. *JMLR.*
- **Erickson, N. et al. (2020).** AutoGluon-Tabular. *arXiv:2003.06505.*
- **Wang, C. et al. (2021).** FLAML: A Fast and Lightweight AutoML Library. *MLSys.*
- PyCaret-Doku: <https://pycaret.gitbook.io>

## Anomaliedetection — Methoden

- **Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).** Isolation Forest. *ICDM.*
- **Schölkopf, B. et al. (2001).** Estimating the Support of a High-Dimensional Distribution
  (One-Class SVM). *Neural Computation.*
- **Breunig, M. et al. (2000).** LOF: Identifying Density-Based Local Outliers. *SIGMOD.*
- **Ruff, L. et al. (2018).** Deep One-Class Classification (Deep SVDD). *ICML.*
- **Ruff, L. et al. (2020).** Deep Semi-Supervised Anomaly Detection (DeepSAD).
  *ICLR / arXiv:1906.02694.* — Repo: <https://github.com/lukasruff/Deep-SAD-PyTorch>
- **Kingma, D. P., & Welling, M. (2014/2019).** Auto-Encoding Variational Bayes / An
  Introduction to Variational Autoencoders. *arXiv:1906.02691.*
- **Li, Z. et al. (2022).** ECOD: Unsupervised Outlier Detection Using Empirical Cumulative
  Distribution Functions. *TKDE.* — parameterfreie Baseline.
- **Breiman, L. (2001).** Random Forests. *Machine Learning.*
- **Kohonen, T. (1990/2001).** The Self-Organizing Map. — SOM-Grundlagen.

## AutoML für Anomaliedetection (Kern)

- **Zhao, Y., Rossi, R., & Akoglu, L. (2021).** Automating Outlier Detection via Meta-Learning
  (MetaOD). *arXiv:2009.10606.* — Repo: <https://github.com/yzhao062/metaod>,
  PyPI: <https://pypi.org/project/metaod/>
- **Zhao, Y. et al. (2021).** Automatic Unsupervised Outlier Model Selection (ELECT).
  *NeurIPS.*
- **Ma, M. Q. et al. (2021/2023).** A Large-scale Study on Unsupervised Outlier Model
  Selection: Do Internal Strategies Suffice? *arXiv:2104.01422.* — interne Metriken vs.
  echte Performance.
- **Goix, N. (2016).** How to Evaluate the Quality of Unsupervised Anomaly Detection
  Algorithms? (Excess-Mass / Mass-Volume). *arXiv:1607.01152.*
- **Marques, H. O. et al.; IREOS/SIREOS-Studie (TKDD 2024).** Enhancing Unsupervised Outlier
  Model Selection: A Study on IREOS Algorithms.

## Outlier Ensembles

- **Aggarwal, C. C., & Sathe, S. (2017).** *Outlier Ensembles: An Introduction.* Springer.
- **Caruana, R. et al. (2004).** Ensemble Selection from Libraries of Models. *ICML.*

## Bibliotheken / Toolkits

- **PyOD** — Zhao, Y., Nasrullah, Z., & Li, Z. (2019). PyOD: A Python Toolbox for Scalable
  Outlier Detection. *JMLR.* — <https://github.com/yzhao062/pyod>
- **DeepOD** — Deep-Learning-AD-Toolkit. <https://github.com/xuhongzuo/DeepOD>
- **combo** — Modellkombination. <https://github.com/yzhao062/combo>
- **MiniSom** — Self-Organizing Maps. <https://github.com/JustGlowing/minisom>
- **Optuna / SMAC3 / Ray Tune** — HPO-Engines (siehe oben).
