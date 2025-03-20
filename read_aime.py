import sys
import csv

def read_aime2024_recaption(line: int):
    with open("aime2024_recaption.csv", "r", encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i == line:
                print(f"[ ID ]: {row['ID']}")
                print(f"[ Problem ]: {row['Problem']}")
                print(f"[ Generated Text ]: {row['Generated Text']}")
                print(f"[ Recap Process ]: {row['Recap Process']}")
                print(f"[ Recap ]: {row['Recap']}")
                print(f"[ Answer ]: {row['Answer']}")
                print(f"[ Solution ]: {row['Solution']}")
                break

if __name__ == "__main__":
    read_aime2024_recaption(int(sys.argv[1]))