from pathlib import Path
from src.data_loader import load_sms_spam_dataset
from src.text_preprocess import batch_clean_text
from src.models import train_test_split_tfidf, train_all_models, results_to_dataframe
from src.bm25_searcher import build_bm25_corpus, search_bm25
import pandas as pd
from time import sleep
from tqdm import tqdm


def show_results(df):
    # Prints stats from kaggle dataset, and then the models' performances
    print("\n=== Dataset statistics (static, does not change) ===")
    print(f"Number of messages parsed: {len(df)} messages.")
    print("Count labeled ham ✔ or spam X:")
    print(df["label"].value_counts())
    print("\nPercentage of messages labeled ham ✔ or spam X (static, does not change):")
    print(df["label"].value_counts(normalize=True) * 100)

    # preprocessing + models
    df["text_clean"] = batch_clean_text(df["text"])
    X_train, X_test, y_train, y_test, vectorizer = train_test_split_tfidf(
        df["text_clean"].tolist(),
        df["target"].tolist(),
    )

    print("\n=== Training info 80/20 split (trained data) ===")
    print(f"Train size: {len(y_train)} messages used to teach the model relationship between spam vs ham")
    print(f"Test size: {len(y_test)} messages used to compute accuracy of model")

    results = train_all_models(X_train, y_train, X_test, y_test)
    results_df = results_to_dataframe(results,X_test,y_test) #also get model-trained results of ham/spam comp to actual

    print("\n=== Model performance (trained data) ===")
    print(results_df.to_string(index=False))

    # who is the best
    best_key, best_model = max(
        results.items(),
        key = lambda item: item[1].f1
    )

    print("\nBest model:", best_model.name)


    print("\n=== Accuracy matrix (trained data) ===")

    #print("[  TrueNegative   |  FalsePositive  ]")
    #print("[  FalseNegative  |  TruePositive  ]")
    #print("TN = guessed 'not spam' which is correct")
    #print("FP = guessed 'spam' which is not correct")
    #print("FN = guessed 'not spam' which is not correct")
    #print("TP = guessed 'spam' which is correct")

    #print(best_res.matrix) #.confusion per models import
    model_array = best_model.confusion
    model_array_df = pd.DataFrame(
        model_array,
        index=["Actual ham", "Actual spam"],
        columns=["Pred ham", "Pred spam"],
    )
    print(model_array_df)
    print("where...")
    print(model_array[0, 0],"   = True Negative = guessed 'not spam' which is correct")
    print(model_array[0, 1],"   = False Positive = guessed 'spam' which is not correct")
    print(model_array[1,0],"    = False Negative = guessed 'not spam' which is not correct")
    print(model_array[1,1],"    = True Positive = guessed 'spam' which is correct")

    print("\n======================================")


def bm25_search(df):
    # Allows the user to input a query to search for dataset match
    # Demonstrates how the model ranks the data
    print("\n=== BM25 search ===")
    bm25 = build_bm25_corpus(df["text"].tolist())

    # User enters search query
    while True:
        query = input("\nEnter a search query (or 'q' to quit BM25 searching): ").strip()

        if query == "":
            print("Look at you, being witty and entering nothing.")
            continue

        if query.lower() in {"q", "quit"}:
            print("Exiting BM25 search.")
            break

        # Return top_n matches (default is 5 for now)
        top_n = 5
        start = 0

        while True:
            top_results = search_bm25(bm25,df["text"].tolist(),query=query,top_n=top_n + start,)[start:start + top_n]

            if not top_results: #No results
                print("No more results for this query!")
                break

            for idx, score, text in top_results: # return index, the BM25 score, and 80 char
                print(f"[{idx}] score={score:.3f}  text={text[:80]}...")

            while True: # New add, allow user to see more results if they want
                choice = input(
                    "\nShow 5 more? (y = more, n/q = new query/quit): "
                ).strip().lower()

                if choice in {"y", "yes", "yep", "ye"}:
                    start += top_n
                    break
                elif choice in {"n", "no", "q", "quit"}:
                    start = 0
                    break
                else:
                    print("Please enter 'y' to see more, or 'n'/'q' to stop.")

            if choice in {"n", "no", "q", "quit"}:
                break


def main():
    #Splash screen. Load the data, welcome the user, wait for first input.

    data_path = Path("data") / "kaggle-dataset.csv" # will be in /data/ directory
    df = load_sms_spam_dataset(data_path)

    print("Hello! Welcome to my Spam Detector Tool.")

    print("My tool loads a Kaggle CSV dataset containing spam or legitimate (ham) messages,")
    print("  and attempts to predict each message's legitimacy.")
    sleep(1)
    print("My app uses Naive Bayes, Logistic Regression, and KNN (K=5) models.")
    sleep(1)
    print("During each run, you may choose a random seed to control data splitting.")
    sleep(1)
    print("You will also be presented with statistics and results.")
    sleep(1)
    print("You may also search messages using BM25 information-retrieval (search query, see BM25 score.")
    sleep(1)
    print("Thanks for checking out my tool!")

    while True:
        print("\n=== SMS Spam Detector Tool - Main Menu ===")
        print("Enter 'r' to show stats and model results.")
        print("Enter 's' to start a BM25 search.")
        print("Enter 'q' to quit.")
        choice = input("Your choice: ").strip().lower()
        print("\n======================================")

        if choice.lower() == "r":
            show_results(df)
        elif choice.lower() == "s" or choice.lower() == "search":
            bm25_search(df)
        elif choice.lower() == "q" or choice.lower() == "quit":
            print("Thanks for checking out my CS410 project! Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 'r', 'b', or 'q'.")

if __name__ == "__main__":
    main()
