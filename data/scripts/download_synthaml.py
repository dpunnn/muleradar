"""
Panduan download SynthAML HI-Large.

SynthAML bukan dari IBM — dari USI Switzerland, Nature Scientific Data 2023.
Dataset tersedia di dua tempat:

1. Kaggle (paling mudah):
   Dataset name: "Synthetic AML Dataset" atau "IBM Transactions for Anti Money Laundering (AML)"
   URL: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
   File yang didownload: HI-Large_Trans.csv (~2GB)

2. Direct dari paper (jika Kaggle tidak tersedia):
   Paper: https://www.nature.com/articles/s41597-022-01915-2
   Cek supplementary data atau GitHub link di paper

Cara download via Kaggle API:
  pip install kaggle
  kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml
  unzip *.zip -d ../raw/

Setelah download, taruh HI-Large_Trans.csv di:
  muleradar/data/raw/HI-Large_Trans.csv

Lalu jalankan:
  python postprocess.py --input ../raw/HI-Large_Trans.csv --output ../processed/transactions.csv

Untuk testing tanpa download (generate mini synthetic):
  python postprocess_test.py
"""

print(__doc__)
