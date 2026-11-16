# Waterlily Take-Home: Predicting 30-Day Readmission for Diabetic Patients

**Author:** Srija Reddy Annam
**Notebook:** `waterlilyassignment_srija.ipynb`
**Data:** 101,766 hospital encounters for 71,518 diabetic patients

## 1. How to run the notebook in Colab

1. Open `waterlilyassignment_srija.ipynb` in Google Colab.
2. In the Files panel (folder icon on the left), upload the three data files to the root folder:
   `diabetic_data.csv`, `diabetes_features.csv` and `IDS_mapping.csv`.
   The notebook reads them from `/diabetic_data.csv`, `/diabetes_features.csv` and `/IDS_mapping.csv`.
3. Choose **Runtime > Run all**. A standard CPU runtime is enough; no GPU is needed.
4. Charts are saved to `reports/figures/` and the trained model to `models/readmission_model.joblib`.

Notes:

- Colab deletes uploaded files when the runtime disconnects, so re-upload them in a new session.
- The API cell starts a server on port 8000. Re-running that cell without restarting the runtime
  prints "address already in use"; the first server is still running and still answers requests.

## 2. Example API request

The notebook starts a FastAPI service with two endpoints: `GET /health` and `POST /predict`.
`/predict` accepts raw encounter fields, using the same names as the CSV columns. Any field left
out is filled with a typical value from the training data (median for numbers, most common value
for categories).

Request (run in a Colab cell with a leading `!`, or in the Colab terminal):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": "[70-80)", "time_in_hospital": 9, "number_inpatient": 4, "number_emergency": 2, "discharge_disposition_id": 22, "insulin": "Up"}'
```

Response:

```json
{
  "readmission_30d_probability": 0.5603,
  "high_risk_flag": true,
  "threshold": 0.1428,
  "fields_defaulted": 37
}
```

For comparison, a low-risk encounter (`{"age": "[50-60)", "time_in_hospital": 2, "number_inpatient": 0}`)
returns a probability of 0.0919 and `high_risk_flag: false`.

An unknown field name returns HTTP 422:

```json
{"detail": "Unknown fields: ['blood_pressure']"}
```

## 3. EDA findings and assumptions

### Findings

- **Size:** 101,766 encounters and 50 columns, covering 71,518 unique patients. One patient has
  40 encounters, so repeat patients are common.
- **Outcome mix:** 53.9% not readmitted, 34.9% readmitted after 30 days, 11.2% readmitted within 30 days.
- **Missing values are coded as `?`:** `weight` is 96.9% missing, `medical_specialty` 49.1%,
  `payer_code` 39.6%, `race` 2.2%, `diag_3` 1.4% and `diag_2` 0.4%.
- **Useless columns:** `examide` and `citoglipton` have a single value in every row.
- **Lab result gotcha:** in `A1Cresult`, 84,748 encounters (83%) say `None`, meaning the test was
  not taken. Pandas reads the text `None` as a missing value by default, so the loader keeps it as
  its own category.
- **Supporting files:** `IDS_mapping.csv` is three lookup tables stacked in one file (8 admission
  types, 30 discharge dispositions, 25 admission sources). The data dictionary describes 23
  medication columns in a single row.
- **Drivers of readmission (charts in `reports/figures/`):**
  - Prior inpatient visits are the strongest driver: the 30-day rate rises from 8.6% with no prior
    inpatient visits to 37.1% with five or more.
  - Discharge destination matters: 9.3% for patients sent home against 27.7% for rehab facilities.
  - Age is a weak driver: most adult age groups sit between 10% and 13%.

### Assumptions

- Encounters ending in death or hospice (discharge codes 11, 13, 14, 19, 20, 21) are removed,
  because those patients cannot be readmitted. This removes 2,423 rows and leaves a cohort of 99,343.
- The target is 1 only for readmission within 30 days (`<30`); `>30` and `NO` are both 0.
  The positive rate in the cohort is 11.4%.
- `weight` and `payer_code` are dropped for missingness. Missing `race` and `medical_specialty`
  become their own category.
- "No record of readmission" is treated as not readmitted, although a patient may have been
  readmitted at a hospital outside the dataset.

## 4. Modeling question and approach

**Question:** will this patient be readmitted within 30 days of discharge?

**Why it matters:** early readmissions are costly, are often preventable, and are a common
hospital quality measure. A risk score at discharge lets care teams focus follow-up calls and
transition support on the patients most likely to return.

### Approach

- **Features:** 46 model features (14 numeric, 32 categorical). Age bands become a number,
  diagnosis codes are collapsed into nine clinical groups, and three summary features are added:
  total prior visits, number of diabetes medications and number of dose changes.
- **Patient-level split:** an 80/20 train/test split by patient, so no patient appears on both
  sides (checked: 0 patients overlap). A second patient-level split inside the training data
  gives a validation set for model selection.

  | Set | Encounters |
  |---|---|
  | Train | 63,807 |
  | Validation | 15,734 |
  | Test | 19,802 |

- **Models compared on validation:**

  | Model | ROC AUC | PR AUC |
  |---|---|---|
  | Logistic regression | 0.665 | 0.219 |
  | Gradient boosting | 0.676 | 0.227 |

  Gradient boosting's gain in PR AUC (+0.0084) was below the 0.01 threshold set in advance, so
  the simpler and more interpretable **logistic regression** was selected.
- **Operating point:** flag the highest-risk 20% of encounters, which is a score cutoff of 0.143.

### Test results (held-out patients, evaluated once)

| Metric | Value |
|---|---|
| ROC AUC | 0.664 |
| PR AUC | 0.220 (base rate 0.112) |
| Brier score | 0.0951 (0.0994 for always predicting the base rate) |
| Encounters flagged | 3,961 of 19,802 |
| Readmissions caught | 38.0% |
| Readmission rate among flagged | 21.3% (11.2% overall) |
| Lift | 1.9x |

Calibration is good: predicted risk tracks the observed readmission rate across ten risk groups,
so the scores can be read as probabilities.

## 5. Errors and limitations

- **Missed readmissions:** of 2,216 readmissions in the test set, 843 were caught and 1,373 were missed.
- **First-time patients are hard to see:** 57.2% of missed readmissions had no prior visits of any
  kind. Caught readmissions averaged 2.66 prior inpatient stays; missed ones averaged 0.37. The
  model leans heavily on visit history.
- **False alarms:** 3,118 of the 3,961 flagged encounters were not readmitted. This is acceptable
  for a low-cost action such as a follow-up call, not for costly interventions.
- **Weak for the oldest patients:** for ages 90-100 the ROC AUC is 0.521, close to chance, even
  though this group is flagged most often (30.3%).
- **Subgroup performance:** ROC AUC is similar for African American (0.661) and Caucasian (0.663)
  patients and for women (0.672) and men (0.656). Patients with missing race are under-flagged
  (9.1% against 20% overall). Small groups, such as ages 20-30 and Hispanic patients, have fewer
  than 400 test encounters, so their scores are noisy.
- **Modest discrimination:** a ROC AUC of 0.66 makes this a ranking aid, not a precise predictor.
- **Old data:** diagnoses use ICD-9 codes, which have since been replaced by ICD-10, and care
  practices have changed. The model would need retraining on current data.
- **Decision support only:** the score should inform clinical judgment, not replace it, and
  should not be used to deny or limit care.
- **API defaults:** filling missing fields with typical values keeps the demo easy to call, but a
  sparse request gets an average-looking score. A real service should require the key fields.

## 6. Three additional business questions

1. **Length of stay at readmission:** predict how long a readmitted patient will stay, to support
   bed and staffing planning and cost forecasting.
2. **HbA1c testing gap:** only 17% of diabetic encounters had HbA1c measured. Find where testing
   is skipped, and compare readmission between tested and untested patients using propensity matching.
3. **High-utilizer segmentation:** cluster at the patient level (some patients have up to 40
   encounters) to target care-management programs.

## 7. What I would do with more time

- Tune both models and check that the model selection holds across several random splits.
- Add features aimed at patients with no visit history, such as specific diagnosis combinations
  and comorbidity scores.
- Choose the cutoff from intervention cost and benefit instead of a fixed top 20%.
- Add confidence intervals to the subgroup results and investigate the weak performance for ages 90+.
- Explain individual predictions (coefficients or SHAP) so clinicians can see the reasons behind a flag.
- Harden the API: a typed request schema, required key fields, tests, a Docker image and drift monitoring.
- Validate on recent ICD-10 data before any real use.
