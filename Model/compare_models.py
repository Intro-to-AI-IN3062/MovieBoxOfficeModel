#combines Linear Regression + RF model table results

import pandas as pd

base = pd.read_csv("Reports/Baseline_Test_Result.csv")
lr = pd.read_csv("Reports/LinearRegression/LR_Test_Result.csv")
rf = pd.read_csv("Reports/RandomForest/RF_Test_Result.csv")

final = pd.concat([base, lr, rf], ignore_index=True)

#Formatting values
final["rmse_test_M"] = (final["rmse_test"] / 1000000).round(2)
final["mae_test_M"] = (final["mae_test"] / 1000000).round(2)

final["r2_test"] = final["r2_test"].round(3)

final = final[["model", "params", "r2_test", "rmse_test_M", "mae_test_M"]]

final.to_csv("Reports/Final_Model_Comparison.csv", index=False)
print(final)