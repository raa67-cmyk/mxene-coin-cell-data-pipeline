# MXene Coin Cell Data Analysis Pipeline

This project contains a modular Python pipeline for processing and analyzing electrochemical coin cell battery data from MXene-based electrodes.

The pipeline automates extraction of:

- Charge/discharge capacity
- Coulombic efficiency
- Energy metrics
- Internal resistance indicators
- dQ/dV analysis
- Capacity fade features
- Automated QC and reports

## Installation

pip install -r requirements.txt

## Running the Pipeline

Place raw cycler files in:

data/raw/

Run:

python pipeline.py

Outputs will be saved in:

outputs/

## Author

Rilwan Adenaike  
Materials Science & Engineering  
Battery Data Analytics & MXene Research
