import csv
from collections import defaultdict

# Read the CSV file and collect PTID and GENOTYPE values
input_file = 'ADNI_Genotype.csv'
output_file = 'processed_genotype_onehot.csv'

ptid_genotype_pairs = []
unique_genotypes = set()

with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ptid = row['PTID']
        genotype = row['GENOTYPE']
        ptid_genotype_pairs.append((ptid, genotype))
        unique_genotypes.add(genotype)

# Sort genotypes to have consistent column order
unique_genotypes = sorted(unique_genotypes)

# Write to new CSV with one-hot encoded genotype columns
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['PTID'] + [f'GENOTYPE_{g}' for g in unique_genotypes]
    writer.writerow(header)

    for ptid, genotype in ptid_genotype_pairs:
        row = [ptid] + [1 if genotype == g else 0 for g in unique_genotypes]
        writer.writerow(row)

output_file
