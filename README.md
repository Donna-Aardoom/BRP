# BRP
Exploring the properties of exoplanets with machine learning


This project uses Random Forest, LightGBM, and XGBoost models to predict exoplanet masses and radii from observable planetary and stellar parameters. The models are trained on exoplanet.eu data combined with Solar System planets and are evaluated using residual analysis and prediction-performance plots.


Planets are removed when:
- Radius uncertainty exceeds 8%
- Mass uncertainty exceeds 25%
- Measurements from chosen or default features are unavailable
- Uncertainty of a number of measurement exceeds the corresponding measurement 
