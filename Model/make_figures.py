import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
from data_prep import clean_data

DATA_PATH = "data/Mojo_budget_update.csv"

LR_RUNS_PATH = "Reports/LinearRegression/LR_Tuning_Runs.csv"
RF_RUNS_PATH = "Reports/RandomForest/RF_Tuning_Runs.csv"

FIG_DIR = "Reports/Figures"

#Exploratory Data Analysis & Tuning result graphs
EDA_CORR_OUT = f"{FIG_DIR}/EDA_CorrelationMatrix.png"
EDA_TARGET_OUT = f"{FIG_DIR}/EDA_WorldwideHistogram.png"
RF_TUNING_OUT = f"{FIG_DIR}/RF_Tuning_RMSE_Val.png"

#WORLDWIDE FIGURES
def save_corr_matrix(df, out_path):
    numeric_df = df.select_dtypes(include=["number"]).copy() #only numeric columns

    corr = numeric_df.corr()
    plt.figure(figsize=(7, 6))

    plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1) #lock color scale (so it doesn't show extreme values)
    plt.colorbar(label="Correlation (-1 to +1)") #legend

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)

    plt.title("Correlation matrix (target + numeric features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_target_hist(df, out_path):
    plt.figure(figsize=(7, 4))
    plt.hist(df["worldwide"].values, bins=40) #40 x-axis intervals

    plt.title("Distribution of worldwide revenue")
    plt.xlabel("worldwide")
    plt.ylabel("count")

    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()

#RFR MODEL FIGURES
def save_rf_tuning_plot(runs_csv, out_path):
    #Load tuning results
    runs = pd.read_csv(runs_csv)
    runs = runs[runs["model"].isin(["Baseline(DummyMean)", "RandomForest"])].copy()

    labels = []
    #build x-axis labels
    for _, row in runs.iterrows():
        if row["model"] == "Baseline(DummyMean)":
            labels.append("Baseline")
        else:
            p = ast.literal_eval(row["params"])
            labels.append(
                f"n={p.get('n_estimators')},depth={p.get('max_depth')},leaf={p.get('min_samples_leaf')}"
            )
            
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(runs)), runs["rmse_val"].values, marker="o")
    plt.xticks(range(len(runs)), labels, rotation=25, ha="right")
    plt.title("Random Forest: validation RMSE across configurations")
    plt.xlabel("configuration")
    plt.ylabel("RMSE (val)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    cleaned_df, X, y, numeric_cols, categorical_cols = clean_data(data) #Clean data again to use for generating figures

    save_corr_matrix(cleaned_df, EDA_CORR_OUT)
    save_target_hist(cleaned_df, EDA_TARGET_OUT)
    save_rf_tuning_plot(RF_RUNS_PATH, RF_TUNING_OUT)

    print("Saving Figures")
    print(EDA_CORR_OUT)
    print(EDA_TARGET_OUT)
    print(RF_TUNING_OUT)


if __name__ == "__main__":
    main()