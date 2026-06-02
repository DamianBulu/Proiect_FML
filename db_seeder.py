import argparse
import pandas as pd

from MedicalKnowledgeRepo import MedicalKnowledgeRepo


def seed_database(start_row: int = 0, max_rows: int = None):
    df = pd.read_csv("./data/pubmedqa.csv")
    total_rows = len(df)
    print(f"CSV loaded: {total_rows} total rows")

    # Slice from start_row
    df = df.iloc[start_row:]
    if max_rows is not None:
        df = df.iloc[:max_rows]

    print(f"Processing rows {start_row} to {start_row + len(df) - 1} ({len(df)} rows)")

    questions = df["Question"].tolist()
    contexts = df["Context"].tolist()
    answers = df["Answer"].tolist()

    texts = [
        f"[QUESTION]\n{questions[i]}\n"
        f"[CONTEXT]\n{contexts[i]}\n"
        f"[ANSWER]\n{answers[i]}"
        for i in range(len(questions))
    ]

    repo = MedicalKnowledgeRepo()
    repo.seed_database(texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the ChromaDB medical knowledge database")
    parser.add_argument("--start-row", type=int, default=0,
                        help="Row index to start from (0-based). Use this to resume after a crash.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum number of rows to process. Omit to process all remaining rows.")
    args = parser.parse_args()

    seed_database(start_row=args.start_row, max_rows=args.max_rows)