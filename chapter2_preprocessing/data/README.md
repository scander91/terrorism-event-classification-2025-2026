# Data Access Instructions

## GTD (Global Terrorism Database)

The GTD is a restricted-access dataset maintained by the National Consortium for the Study of Terrorism and Responses to Terrorism (START) at the University of Maryland.

### How to obtain:
1. Visit: https://www.start.umd.edu/gtd/contact/
2. Fill in the data request form
3. Specify academic/research use
4. Typical approval time: 1-5 business days
5. Once approved, download the CSV file
6. Place it as: `chapter2_preprocessing/data/globalterrorismdb.csv`

### Expected file:
- Filename: `globalterrorismdb_0718dist.xlsx` or similar
- Rows: ~209,706 incidents
- Columns: 135 features
- Coverage: 1970-2020

## Crime Datasets (Chapter 3 cross-domain)

These are publicly available:

- **SF Crime Data**: https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783
- **LA Crime Data**: https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8

## Arabic NLP Benchmarks (Chapter 5)

| Dataset | Source | Access |
|---------|--------|--------|
| MADAR | Bouamor et al. 2019 | Request from authors |
| NADI 2020 | Abdul-Mageed et al. 2020 | https://github.com/Wikipedia-based-ADI |
| ASTD | Nabil et al. 2015 | https://github.com/mahmoudnabil/ASTD |
| ArSAS | AbdelRahim & Magdy 2018 | Request from authors |
| ASAD | Alharbi et al. 2020 | https://github.com/basharalhafni/ASAD |
| LABR | Aly & Atiya 2013 | https://github.com/mohamedadaly/LABR |
| OSACT4 | Mubarak et al. 2020 | Shared task data |
| Adult Content | Mubarak et al. 2021 | Request from authors |
| SemEval-2016 | Kiritchenko et al. 2016 | https://alt.qcri.org/semeval2016/task7/ |
| Alomari 2017 | Alomari et al. 2017 | Request from authors |
| ArSentiment | Various | Aggregated from above |

## Preprocessed Data

After running the Chapter 2 pipeline, the preprocessed dataset will be saved at:
`chapter2_preprocessing/data/gtd_preprocessed.csv`

This file is used as input by Chapters 3, 4, and 5.

## Important Note

**Do NOT upload any dataset files to GitHub.** The `.gitignore` is configured to exclude CSV, Excel, and other data files. Users must obtain data independently following the instructions above.
