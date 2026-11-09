# Customer Churn Prediction & Expected-Value Retention

Predicts which telecom customers will churn, turns the scores into calibrated
probabilities, and decides **who is worth a retention offer** by weighing churn
risk against how much revenue each customer brings in. Served as a FastAPI app
in Docker.

## Why expected value?

A retention budget can only cover some customers. Ranking by churn probability
alone spends it on likely churners even when they pay very little. This project
ranks by the expected net gain of an offer instead:

```
EV = P(churn) × save_rate × CLTV − offer_cost
```

`P(churn)` has to be a real probability for this to work, which is why the
Random Forest is calibrated with isotonic regression.

## Results

IBM Telco Customer Churn dataset: 7,032 customers, 26.6% churn rate.
All numbers are out-of-fold from 5-fold stratified cross-validation
(`reports/metrics.json` is regenerated on every run).

| | Value |
|---|---|
| Random Forest ROC-AUC | 0.847 ± 0.004 |
| Logistic Regression ROC-AUC (baseline) | 0.845 ± 0.003 |
| Brier score, raw Random Forest | 0.155 |
| Brier score, after isotonic calibration | 0.135 |
| Net revenue saved, contacting top 20% ranked by probability | $402K |
| Net revenue saved, contacting top 20% ranked by expected value | $439K (**+9.0%**) |

The uplift depends on the campaign budget (+4% to +9% for budgets of 10–40%).
Campaign assumptions live at the top of `src/value.py`: 24-month CLTV horizon,
1% monthly discount, 30% save rate, $50 per offer.

## Run it

```bash
pip install -r requirements-dev.txt
python -m src.train          # downloads data, benchmarks, calibrates, saves model
pytest -q                    # 8 tests
uvicorn src.api:app --reload # docs at http://localhost:8000/docs
```

With Docker (the image trains the model during the build):

```bash
docker build -t churn-retention .
docker run -p 8000:8000 churn-retention
```

## API

`POST /predict` scores one customer. Only `tenure`, `MonthlyCharges` and
`TotalCharges` are required; other fields have defaults.

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 95, "TotalCharges": 190}'
```

```json
{"churn_probability": 0.727, "customer_value": 2018.12,
 "expected_value_of_offer": 390.18, "send_offer": true}
```

`POST /predict/batch` scores a list and returns it sorted by expected value.
`GET /health` is a liveness check.

## Layout

```
src/data.py    load and clean the dataset
src/model.py   Logistic Regression, Random Forest, isotonic-calibrated RF
src/value.py   CLTV, expected value, campaign evaluation
src/train.py   cross-validated benchmark and final model
src/api.py     FastAPI service
tests/         API and value-logic tests
```
