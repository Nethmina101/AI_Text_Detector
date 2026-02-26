import pandas as pd
import os

def clean_dataset(input_csv="data/AI_Human.csv", output_csv="data/AI_Human_Cleaned.csv"):
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}")
        return

    print(f"Loading '{input_csv}'...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    original_count = len(df)
    print(f"Original Row Count: {original_count}")

    if 'text' not in df.columns or 'generated ' not in df.columns:
        print(f"Error: Missing required columns 'text' or 'generated ' in dataset.")
        print(f"Found columns: {list(df.columns)}")
        return

    # 1. Drop completely empty rows or rows missing text/labels
    df = df.dropna(subset=['text', 'generated '])
    dropna_count = len(df)
    print(f"After dropping missing text/labels: {dropna_count} (Removed {original_count - dropna_count})")

    # 2. Convert text to string and drop perfect duplicates based on the 'text' column
    df['text'] = df['text'].astype(str)
    
    # We strip whitespace to catch duplicates that just have extra spaces at the end
    df['stripped_text'] = df['text'].str.strip()
    df = df.drop_duplicates(subset=['stripped_text'])
    df = df.drop(columns=['stripped_text']) # Remove helper column

    dedup_count = len(df)
    print(f"After dropping duplicate text rows: {dedup_count} (Removed {dropna_count - dedup_count})")

    # 3. Handle Extreme Lengths (Very Short or Very Long)
    # Exclude rows where the text is less than 50 characters (not enough stylometry data)
    # Exclude rows where text is MASSIVE (e.g., > 15000 chars) which crashed your RAM initially
    df['text_len'] = df['text'].str.len()
    df = df[(df['text_len'] >= 50) & (df['text_len'] <= 15000)]
    df = df.drop(columns=['text_len'])
    
    final_count = len(df)
    print(f"After removing texts that are too short (<50) or massive (>15k): {final_count} (Removed {dedup_count - final_count})")
    
    total_removed = original_count - final_count
    
    # Save the cleaned dataset
    try:
        df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved cleaned dataset to '{output_csv}'")
        print(f"Total Rows Removed: {total_removed} ({(total_removed/original_count)*100:.1f}%)")
        print("IMPORTANT: You must now update src/train.py to use this new file or rename it over the old one.")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    clean_dataset()
