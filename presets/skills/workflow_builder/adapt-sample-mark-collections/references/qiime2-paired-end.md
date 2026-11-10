# Example: Import Filtered Paired-End Reads into QIIME 2

## Goal

Use the filtered R1 and R2 files made by every sample workflow in one QIIME 2 import run.

The QIIME tool has two inputs:

- `manifest` receives a TSV file that lists every sample and its R1/R2 paths.
- `reads_root` makes the collected FASTQ files available inside the QIIME container at `/data/reads`.

## Step 1: Save the Two Files in the Sample Workflow

Connect the filtered outputs to two `sample_mark` nodes:

```text
filtered R1 -> sample_mark with mark_name = primer_trimmed_r1
filtered R2 -> sample_mark with mark_name = primer_trimmed_r2
```

Run this workflow successfully for every sample that should be included in QIIME.

## Step 2: Collect the Files in a Project-Level Workflow

Add `sample_mark_collection` with:

```text
mark_names = ["primer_trimmed_r1", "primer_trimmed_r2"]
```

## Step 3: Connect the Nodes

Use these connections:

```text
collection_manifest
  -> bind_param
  -> Python or Bash CODE node
  -> select_file_in_folder(file_name = "pe_manifest.tsv")
  -> QIIME.manifest

mount_collection
  -> QIIME.reads_root
```

For Python, set the bind parameter to `--collection-manifest`. For Bash, use a short parameter such as `-m`.

## Step 4: Generate the QIIME TSV

Ask the CODE node to write `pe_manifest.tsv`. Its first line must be:

```text
sample-id\tforward-absolute-filepath\treverse-absolute-filepath
```

For each sample:

- Find the R1 entry whose `mark_name` is `primer_trimmed_r1`.
- Find the R2 entry whose `mark_name` is `primer_trimmed_r2`.
- Use `sample_name` for the first column.
- Write `/data/reads/<mount_name>` for each FASTQ path.

For example:

```text
sample-id    forward-absolute-filepath                                      reverse-absolute-filepath
S1           /data/reads/S1/primer_trimmed_r1/sample_R1.fastq.gz            /data/reads/S1/primer_trimmed_r2/sample_R2.fastq.gz
```

Do not copy FASTQ files in the CODE node. Do not use `source_path`. Do not hard-code a fixed list such as S1 through S4;
read all sample names from the JSON.

The QIIME command may not mention `reads_root`. This is expected. Connecting `mount_collection` to `reads_root` tells
the backend to mount the FASTQ files at `/data/reads`, which makes the paths in `pe_manifest.tsv` valid.
