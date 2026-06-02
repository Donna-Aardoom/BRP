from matplotlib import cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mass_models
import lime
import lime.lime_tabular

# Load datasets including error columns
dataset_all = "data/exoplanet.eu_catalog_20-01-26_15_03_11.csv"
cat_solar = 'data/solar_system_planets_catalog.csv'

dataset = mass_models.load_dataset_errors(remove_bad_planets=False)

# otegi threshold data selection
selection_uncertainty = (
    (dataset['mass_error'] / dataset['mass'] < 0.25) &
    (dataset['radius_error'] / dataset['radius'] < 0.08)
)
print("Number of removed planets due to Otegi uncertainty selection: ", len(dataset[~selection_uncertainty])  )
dataset = dataset[selection_uncertainty]

#remove K2-123 b 
planets_to_remove = ['K2-123 b']
dataset = dataset.drop(index=planets_to_remove, errors='ignore')

def plot_predicted_vs_true(
    rf_results,
    lgbm_results,
    xgb_results,
    log_scale=True,
    relative_residuals=True):

    rf_model, rf_pred, rf_values, rf_sets, rf_metrics = rf_results
    lgbm_model, lgbm_pred, lgbm_values, lgbm_sets, lgbm_metrics = lgbm_results
    xgb_model, xgb_pred, xgb_values, xgb_sets, xgb_metrics = xgb_results

    y_test_rf = np.asarray(rf_sets[3])
    y_test_lgbm = np.asarray(lgbm_sets[3])
    y_test_xgb = np.asarray(xgb_sets[3])

    rf_pred = np.asarray(rf_pred)
    lgbm_pred = np.asarray(lgbm_pred)
    xgb_pred = np.asarray(xgb_pred)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1.7]}
    )

    models = [
        ("Random Forest", y_test_rf, rf_pred, rf_metrics),
        ("LightGBM", y_test_lgbm, lgbm_pred, lgbm_metrics),
        ("XGBoost", y_test_xgb, xgb_pred, xgb_metrics)
    ]

    markers = ['*', 'o', '^']

    all_true = []
    all_pred = []

    for (name, y_true, y_pred, metrics), marker in zip(models, markers):
        label = (
            rf"{name}: "
            rf"$\epsilon$={metrics['epsilon']:.3f}, "
            rf"$R^2_{{\rm train}}$={metrics['r2_train']:.3f}, "
            rf"$R^2_{{\rm test}}$={metrics['r2_test']:.3f}"
        )

        ax1.scatter(
            y_true,
            y_pred,
            alpha=0.7,
            label=label,
            marker=marker,
            s=30
        )

        if relative_residuals:
            residuals = (y_pred - y_true) / y_true
        else:
            residuals = y_true - y_pred

        ax2.scatter(
            y_true,
            residuals,
            alpha=0.7,
            label=name,
            marker=marker,
            s=30
        )

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    min_value = min(all_true.min(), all_pred.min())
    max_value = max(all_true.max(), all_pred.max())

    ax1.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect prediction"
    )

    ax2.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=2
    )

    if log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

    ax1.set_ylabel(r"Predicted mass [$M_\oplus$]")
    ax1.set_title("Mass Prediction Comparison: RF vs LightGBM vs XGBoost")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    ax2.set_xlabel(r"True mass [$M_\oplus$]")

    if relative_residuals:
        ax2.set_ylabel("Relative residual")
    else:
        ax2.set_ylabel(r"Residual [$M_\oplus$]")

    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()

def plot_LIME_mass_predictions(regr, train_test_sets,
                               planets=None,
                               feature_name=None,
                               model_name="Model"):
    """
    Compute and plot LIME explanations for mass predictions.

    The model predicts log10 planetary mass, so LIME weights are in
    log10(M/M_earth) space.
    """

    if planets is None:
        planets = []

    X_train, X_test, y_train, y_test = train_test_sets

    if feature_name is None:
        feature_name = list(X_train.columns)

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_name)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_name)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)

    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_name,
        class_names=["log_mass"],
        verbose=True,
        mode="regression"
    )

    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=feature_name)
        return regr.predict(x_df)

    # Use the same default planets as the radius LIME plot
    if not planets:
        default_names = [
            "TOI-561 b",
            "HATS-35 b",
            "CoRoT-13 b",
            "Kepler-75 b",
            "WASP-17 b",
            "Kepler-20 Ac"
        ]

        for name in default_names:
            matches = np.where(X_test.index == name)[0]
            if len(matches) > 0:
                planets.append(matches[0])

        if len(planets) == 0:
            print("None of the default LIME planets are in the test set.")
            planets = list(range(min(6, len(X_test))))

    planets = planets[:6]

    fig, axs = plt.subplots(
        3, 2,
        constrained_layout=True,
        figsize=(15, 45)
        

    
    )

    axs = axs.flatten()

    exp = None

    for j, planet in enumerate(planets):
        x_planet = X_test.iloc[planet]
        true_mass = y_test.iloc[planet]
        true_log_mass = np.log10(true_mass)

        model_log_mass = float(regr.predict(X_test.iloc[[planet]])[0])
        model_mass = 10 ** model_log_mass

        exp = explainer.explain_instance(
            x_planet.values,
            predict_fn,
            num_features=len(feature_name)
        )

        lime_log_mass = float(np.ravel(exp.local_pred)[0])
        lime_mass = 10 ** lime_log_mass

        print("\nPlanet:", X_test.index[planet])
        print("True mass:", true_mass)
        print("Model mass:", model_mass)
        print("True log mass:", true_log_mass)
        print("LIME log mass:", lime_log_mass)

        exp_list = exp.as_list()
        vals = [x[1] for x in exp_list]
        names = [
            x[0].replace("<=", r"$\leq$")
                .replace("_", " ")
                .replace(".00", "")
                .replace("<", "$<$")
                .replace(">", "$>$")
            for x in exp_list
        ]

        vals.reverse()
        names.reverse()

        colors = ["C2" if x > 0 else "C3" for x in vals]
        pos = np.arange(len(exp_list)) + 0.5

        axs[j].get_yaxis().set_visible(False)
        axs[j].set_xlabel(r"Weight in $\log_{10}(M/M_\oplus)$")
        axs[j].set_ylabel("Feature")
        axs[j].set_title(X_test.index[planet], loc="right")

        rects = axs[j].barh(
            pos,
            vals,
            align="center",
            color=colors,
            alpha=0.5
        )

        for i, rect in enumerate(rects):
            axs[j].text(
                axs[j].get_xlim()[0] + 0.03,
                rect.get_y() + 0.2,
                str(names[i])
            )

        textstr = "\n".join((
            r"True mass=%.2f$M_\oplus$" % (true_mass,),
            f"{model_name} mass={model_mass:.2f}$M_\\oplus$",
            r"LIME mass=%.2f$M_\oplus$" % (lime_mass,)
        ))

        axs[j].text(
            0.68,
            0.10,
            textstr,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.8
            },
            transform=axs[j].transAxes
        )

    for k in range(len(planets), len(axs)):
        axs[k].axis("off")

    # Extra plot: mass-radius relation with LIME planets circled
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Mass-radius relation of the test set with LIME predicted planets")

    if "log_temp_eq" in X_test.columns:
        size = 10 ** X_test["log_temp_eq"].values
    else:
        size = np.ones(len(X_test))

    plt.scatter(
        y_test.values,                       # true mass on x-axis
        10 ** X_test["log_radius"].values,   # radius on y-axis
        c=size,
        cmap=cm.magma_r
    )

    plt.colorbar(label=r"Equilibrium temperature (K)")
    plt.xlabel(r"Mass ($M_\oplus$)")
    plt.ylabel(r"Radius ($R_\oplus$)")

    for planet in planets:
        plt.plot(
            y_test.values[planet],
            10 ** X_test.iloc[planet]["log_radius"],
            "o",
            mfc="none",
            ms=12,
            label=X_test.index[planet]
        )

    if len(planets) > 0:
        plt.legend()

    return exp


if __name__ == "__main__":
    #### FULL DATA ###
    print("Number of planets after cleaning: ", len(dataset) )
    
    # Train all three mass-prediction models.
    rf_results = mass_models.random_forest_mass(dataset, fit=True)
    lgbm_results = mass_models.lightgbm_mass(dataset, fit=True)
    xgb_results = mass_models.xgboost_mass(dataset, fit=True)

    # Plot predicted mass versus true mass for all three models.
    plot_predicted_vs_true(
        rf_results,
        lgbm_results,
        xgb_results,
        log_scale=True
    )

    # # Plot LIME explanations for all models
    # plot_LIME_mass_predictions(
    #     lgbm_results[0],
    #     lgbm_results[3],
    #     model_name="LightGBM"
    # )       
    # plot_LIME_mass_predictions( 
    #     rf_results[0],
    #     rf_results[3],
    #     model_name="Random Forest"
    # )

    # plot_LIME_mass_predictions(
    #     xgb_results[0],
    #     xgb_results[3],
    #     model_name="XGBoost"
    # )

    # ### MOUSAVI SPLIT ###

    # # Train all three mass-prediction models on the Mousavi small and giant subsets.
    # dataset_small_mousavi = dataset[dataset["mass"] < 52.48]
    # dataset_giants_mousavi = dataset[dataset["mass"] >= 52.48]
    # print("Number of small planets: ", len(dataset_small_mousavi))
    # print("Number of giant planets: ", len(dataset_giants_mousavi))

    # rf_results_small_ = mass_models.random_forest_mass(dataset_small_mousavi, fit=True)
    # rf_results_giants_ = mass_models.random_forest_mass(dataset_giants_mousavi, fit=True)
    # lgbm_results_small_ = mass_models.lightgbm_mass(dataset_small_mousavi, fit=True)
    # lgbm_results_giants_ = mass_models.lightgbm_mass(dataset_giants_mousavi, fit=True)
    # xgb_results_small_ = mass_models.xgboost_mass(dataset_small_mousavi, fit=True)
    # xgb_results_giants_ = mass_models.xgboost_mass(dataset_giants_mousavi, fit=True)

    # # plot predicted mass versus true mass for all three models on the Mousavi small and giant subsets.
    # plot_predicted_vs_true(
    #     rf_results_small_,
    #     lgbm_results_small_,
    #     xgb_results_small_,
    #     log_scale=True
    # )   
    # plot_predicted_vs_true(
    #     rf_results_giants_,
    #     lgbm_results_giants_,
    #     xgb_results_giants_,
    #     log_scale=True
    # )   

    # #plot LIME explanations for all three models on the Mousavi small and giant subsets.
    # plot_LIME_mass_predictions(
    #     lgbm_results_small_[0],
    #     lgbm_results_small_[3], 
    #     planets=[0, 1, 2, 3, 4, 5]
    # )
    # plot_LIME_mass_predictions(     
    #     rf_results_small_[0],
    #     rf_results_small_[3],   
    #     planets=[0, 1, 2, 3, 4, 5]
    # )

    # plot_LIME_mass_predictions(
    #     xgb_results_small_[0],  
    #     xgb_results_small_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )       

    # plot_LIME_mass_predictions(
    #     lgbm_results_giants_[0],    
    #     lgbm_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )
    # plot_LIME_mass_predictions( 
    #     rf_results_giants_[0],
    #     rf_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   
    # plot_LIME_mass_predictions(
    #     xgb_results_giants_[0], 
    #     xgb_results_giants_[3],
    #     planets=[0, 1, 2, 3, 4, 5]
    # )   









    # ### OTEGI SPLIT ###

    # # Train three mass prediction models on the Otegi small, intermediate, and giant subsets.
    # dataset_small = dataset[dataset["mass"] < 4.4]
    # dataset_intermediate = dataset[(dataset["mass"] >= 4.4) & (dataset["mass"] < 52.48)]
    # dataset_giants = dataset[dataset["mass"] >= 52.48]
    # print("Number of small planets: ", len(dataset_small))
    # print("Number of intermediate planets: ", len(dataset_intermediate))
    # print("Number of giant planets: ", len(dataset_giants))
    # rf_results_small = mass_models.random_forest_mass(dataset_small, fit=True)
    # rf_results_intermediate = mass_models.random_forest_mass(dataset_intermediate, fit=True)
    # rf_results_giants = mass_models.random_forest_mass(dataset_giants, fit=True)
    # lgbm_results_small = mass_models.lightgbm_mass(dataset_small, fit=True)
    # lgbm_results_intermediate = mass_models.lightgbm_mass(dataset_intermediate, fit=True)
    # lgbm_results_giants = mass_models.lightgbm_mass(dataset_giants, fit=True)
    # xgb_results_small = mass_models.xgboost_mass(dataset_small, fit=True)   
    # xgb_results_intermediate = mass_models.xgboost_mass(dataset_intermediate, fit=True)
    # xgb_results_giants = mass_models.xgboost_mass(dataset_giants, fit=True)

    # # plot predicted mass versus true mass for all three models on the Otegi subsets.
    # plot_predicted_vs_true(
    #     rf_results_small,
    #     lgbm_results_small,
    #     xgb_results_small,
    #     log_scale=True
    # )
    # plot_predicted_vs_true(
    #     rf_results_intermediate,
    #     lgbm_results_intermediate,
    #     xgb_results_intermediate,
    #     log_scale=True
    # )
    # plot_predicted_vs_true(
    #     rf_results_giants,
    #     lgbm_results_giants,
    #     xgb_results_giants,
    #     log_scale=True
    # )

    # # #plot LIME explanations for all three models on the Mousavi small and giant subsets.
    # # plot_LIME_mass_predictions(
    # #     lgbm_results_small_[0],
    # #     lgbm_results_small_[3], 
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )
    # # plot_LIME_mass_predictions(     
    # #     rf_results_small_[0],
    # #     rf_results_small_[3],   
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )

    # # plot_LIME_mass_predictions(
    # #     xgb_results_small_[0],  
    # #     xgb_results_small_[3],
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )       

    # # plot_LIME_mass_predictions(
    # #     lgbm_results_intermediate[0],
    # #     lgbm_results_intermediate[3],
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )   

    # # plot_LIME_mass_predictions(
    # #     rf_results_intermediate[0],     
    # #     rf_results_intermediate[3],
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )       

    # # plot_LIME_mass_predictions(
    # #     xgb_results_intermediate[0],    
    # #     xgb_results_intermediate[3],
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )   

    # # plot_LIME_mass_predictions(
    # #     lgbm_results_giants_[0],    
    # #     lgbm_results_giants_[3],
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )
    # # plot_LIME_mass_predictions( 
    # #     rf_results_giants_[0],
    # #     rf_results_giants_[3],
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )   
    # # plot_LIME_mass_predictions(
    # #     xgb_results_giants_[0], 
    # #     xgb_results_giants_[3],
    # #     planets=[0, 1, 2, 3, 4, 5]
    # # )   


    plt.show()