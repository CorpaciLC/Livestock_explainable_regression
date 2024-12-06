# User Readiness Assessment using Explainable Regression

This repository contains the code and data for a study on assessing user readiness for technology adoption using explainable regression models.

Repository Contents
- data/: Contains the survey data used for training and testing the models.
  - X, y (_train, _test) <- files with saved split (for consistency between lime/shap/pdp)
  - labelled_kmeans_data_pig_poultry_last_verified_by_thomas_and_ildiko <- file with labels from previous paper. Agreed with Thomas B. and Ildiko
- src/: Contains source files and relevant notebooks.
- models/: Saved Random Forest Model with 80% accuracy.
- docs/: Contains articles on the topic
- images/: Generated images

````{verbatim}
@article{mallinger2024breaking,
  title={Breaking the Barriers of Technology Adoption: Explainable AI for Requirement Analysis and Technology Design in Smart Farming},
  author={Mallinger, Kevin and Corpaci, Luiza and Neubauer, Thomas and Tik{\'a}sz, Ildik{\'o} E and Goldenits, Georg and Banhazi, Thomas},
  journal={Smart Agricultural Technology},
  pages={100658},
  year={2024},
  publisher={Elsevier}
}

````