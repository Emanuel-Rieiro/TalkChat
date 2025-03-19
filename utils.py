import csv

def read_tsv(file_path):
    """Reads a TSV file and returns the data as a list of dictionaries."""
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

# Example usage
# parent_folder = 'recordings/cv-corpus-18.0-delta-2024-06-14/es'
# file_path = f"{parent_folder}/other.tsv"  # Replace with your TSV file path
# data = read_tsv(file_path)