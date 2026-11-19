# City Signal

City Signal is a Python and PyTorch project that predicts the next hour of **reported NYPD complaint counts** on a 12 × 12 grid over New York City. The interactive dashboard compares observed counts, CNN forecasts, and the same hour last week. You can select a date or map cell and export an hour as CSV.

## Accuracy

The model used 8.89 million geocoded complaints from 2006 through 2023. It trained on the first 80% of hours and was evaluated on the last **31,556 hours**. Lower mean absolute error (MAE) is better.

| Mean absolute error | CNN | Same hour last week |
| --- | ---: | ---: |
| Complaints per active grid cell per hour | 0.537 | 0.669 |
| Citywide complaints per hour | 7.57 | 10.13 |

The CNN reduced per-cell MAE by about **20%** relative to this baseline. Three training epochs gave lower holdout error than four (0.537 vs. 0.543).

## Limitations

- The dataset measures complaints recorded by the NYPD. Reporting and enforcement practices affect those counts.
- Each cell spans about 4 km. Records with unusable dates, times, or coordinates are excluded.
- Evaluation is retrospective and uses earlier counts indexed by occurrence time. Some records may have reached the NYPD after the forecast hour. Performance on live data and after 2023 is unknown.

## Run locally

With Python 3.12, run the following commands. The pipeline downloads about 647 MB of Parquet data and generates the dashboard results locally.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python model_pipeline.py
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

Open [localhost:8000](http://localhost:8000) for the last 28 days of the holdout. The model, hourly counts, and dashboard data are saved in `artifacts/`. For an epoch comparison, run `.venv/bin/python model_pipeline.py --epochs 10 --patience 3 --output artifacts/epoch-comparison`.

Sources: [NYPD complaint dataset on Hugging Face](https://huggingface.co/datasets/ehong/nypd_complaint_data_historic_ds) · [NYC borough boundaries](https://services7.arcgis.com/HDgMIJDCbHtnomY9/ArcGIS/rest/services/NYC_Borough_Boundaries/FeatureServer/0).
