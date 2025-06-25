import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

#Accuracy = Agree / Total
#Precision = Agree / #Pred > 0
#Recall = Agree / #Actual > 0
#F1 = 2 * Rec * Prec / (Rec + Prec)

# A Genetic test is a test that checks for a condition that is known to be caused genetically. Includes karyotypes, microarrays

#Train/Validation/Test + Split

# Load the tab-separated CSV
df = pd.read_csv("claude_TestsList_with_results.csv")


# Convert columns to string (or int) if needed
df["label_normalized"] = df["label_normalized"].astype(str).str.strip()
df["Results"] = df["Results"].astype(str).str.strip()

# Compare values and count matches
matches = df["label_normalized"] == df["Results"]
num_matches = matches.sum()
total = len(df)

false_positives_mask = (df["label_normalized"] != df["Results"]) & (df["Results"].astype(int) == 1)
false_positives = df[false_positives_mask]
false_positives.to_csv("claude_false_positives.csv", index=False)

false_negatives_mask = (df["label_normalized"] != df["Results"]) & (df["Results"].astype(int) == 0)
false_negatives = df[false_negatives_mask]
false_negatives.to_csv("claude_false_negatives.csv", index=False)

print("-------- CLAUDE -------")
print(f"Number of matching values: {num_matches}")
print(f"Total number of rows: {total}")
print(f"Match rate: {num_matches / total:.2%}")
df.label_normalized = pd.to_numeric(df.label_normalized)
df.Results = pd.to_numeric(df.Results)
[precision, recall, f_score, support] = precision_recall_fscore_support(df["label_normalized"], df["Results"], average='binary')
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-Score: " + str(f_score))

# LLAMA --------------

# Load the tab-separated CSV
df = pd.read_csv("llama_TestsList_with_results.csv")


# Convert columns to string (or int) if needed
df["label_normalized"] = df["label_normalized"].astype(str).str.strip()
df["Results"] = df["Results"].astype(str).str.strip()

incoherent_mask = (df["Results"].astype(int) == 2)
incoherent = df[incoherent_mask]
incoherent.to_csv("llama_incoherent.csv", index=False)

no_unknowns_mask = df["Results"].astype(int) != 2
df = df[no_unknowns_mask]

# Compare values and count matches
matches = df["label_normalized"] == df["Results"]
num_matches = matches.sum()
total = len(df)

false_positives_mask = (df["label_normalized"] != df["Results"]) & (df["Results"].astype(int) == 1)
false_positives = df[false_positives_mask]
false_positives.to_csv("llama_false_positives.csv", index=False)

false_negatives_mask = (df["label_normalized"] != df["Results"]) & (df["Results"].astype(int) == 0)
false_negatives = df[false_negatives_mask]
false_negatives.to_csv("llama_false_negatives.csv", index=False)

print("-------- LLAMA -------")
print(f"Number of matching values: {num_matches}")
print(f"Total number of rows: {total}")
print(f"Match rate: {num_matches / total:.2%}")
df.label_normalized = pd.to_numeric(df.label_normalized)
df.Results = pd.to_numeric(df.Results)
[precision, recall, f_score, support] = precision_recall_fscore_support(df["label_normalized"], df["Results"], average='binary')
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-Score: " + str(f_score))