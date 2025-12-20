#data cleanup (now modularised)
import pandas as pd

def clean_data(df):
    df = df.copy()

    runtime_text = df["run_time"].fillna("")
    hours = runtime_text.str.extract(r"(\d+)\s*hr", expand=False).fillna(0).astype(int)
    minutes = runtime_text.str.extract(r"(\d+)\s*min", expand=False).fillna(0).astype(int)
    df["run_time_minutes"] = hours * 60 + minutes

    df = df.drop(columns=[
        "movie_id", "title", "trivia", "html",
        "release_date", "run_time",
        "distributor", "director", "writer", "producer",
        "composer", "cinematographer",
        "main_actor_1", "main_actor_2", "main_actor_3", "main_actor_4"
    ], errors="ignore")

    df["worldwide"] = pd.to_numeric(df["worldwide"], errors="coerce")
    df = df.dropna(subset=["worldwide"])

    y = df["worldwide"]
    X = df.drop(columns=["worldwide", "domestic", "international"], errors="ignore")

    categorical_cols = ["genre_1", "genre_2", "genre_3", "genre_4", "mpaa"]
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    return df, X, y, numeric_cols, categorical_cols
