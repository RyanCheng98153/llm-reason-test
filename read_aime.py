import sys
import csv

def read_aime2024_recaption(line: int):
    with open("aime2024_recaption.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i == line:
                print(f"[ ID ]: \n{row['ID']}")
                print(f"[ Problem ]: \n{row['Problem']}")
                print(f"[ Ground Truth ]: \n{row['Ground Truth']}")
                print(f"[ Recap Process ]: \n{row['Recap Process']}")
                print(f"[ Recap ]: \n{row['Recap']}")
                print(f"[ Recap Answer ]: \n{row['Answer']}")
                print(f"[ Prompt ]: \n{row['Prompt']}")
                print(f"[ Generated Text ]: \n{row['Generated Text']}")
                print(f"[ Solution ]: \n{row['Solution']}")
                break

if __name__ == "__main__":
    read_aime2024_recaption(int(sys.argv[1]))