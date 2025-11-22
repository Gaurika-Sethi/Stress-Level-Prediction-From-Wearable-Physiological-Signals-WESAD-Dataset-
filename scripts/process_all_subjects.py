from extract_labels import extract_labels
from extract_wrist_signals import extract_wrist_signals
from clean_signals import clean_signals
from extract_features import extract_features

SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "S10", "S11", "S13", "S14", "S15", "S16", "S17"
]

if __name__ == "__main__":
    for S in SUBJECTS:
        print(f"\n==================== Processing {S} ====================")
        extract_labels(S)
        extract_wrist_signals(S)
        clean_signals(S)
        extract_features(S)
    print("\n[OK] All subjects processed!")