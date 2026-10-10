# Travel Insurance Prediction — Streamlit app
### Business Problem

Travel insurance companies may waste marketing resources targeting customers who are unlikely to purchase insurance. This project uses a Random Forest classification model to predict customer purchase behaviour based on demographic, financial, and travel-related attributes, helping businesses improve marketing targeting and customer acquisition efficiency.


This is a Streamlit user interface for the Random Forest `GridSearchCV` model in `TRAVEL_INS(1).IPYNB`.

## Important: export your actual trained model first

The uploaded notebook contains the training workflow, but it does not include a saved `travel_insurance_model.pkl` artifact. The UI will show a helpful message until you export the model from the notebook.

### 1. Export the model from your notebook

Run this cell **after** `grid_search_rf.fit(X_train, y_train)` has completed. The notebook maps the following columns to numbers before training:

- `Employment Type`: Government Sector = 1; Private Sector/Self Employed = 0
- `FrequentFlyer`: Yes = 1; No = 0
- `EverTravelledAbroad`: Yes = 1; No = 0
- `ChronicDiseases` is already 0/1
- The target is `TravelInsurance` (0/1)
- `Unnamed: 0` and `GraduateOrNot` are dropped

```python
import joblib
from pathlib import Path

# Save the same fitted model object used in your notebook.
joblib.dump(grid_search_rf, "travel_insurance_model.pkl")

# Save expected input feature names for reference.
joblib.dump(list(X_train.columns), "travel_insurance_features.pkl")
print("Model exported:", Path("travel_insurance_model.pkl").resolve())
```

Copy `travel_insurance_model.pkl` into this app folder, beside `app.py`. The app expects these features in this exact order:

```text
Age
Employment Type
AnnualIncome
FamilyMembers
ChronicDiseases
FrequentFlyer
EverTravelledAbroad
```

The model in the notebook was trained with `Employment Type`, `FrequentFlyer`, and `EverTravelledAbroad` numerically encoded as shown above. The UI reproduces these mappings.

## 2. Run locally

Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Your browser should open a local page, usually `http://localhost:8501`.

## 3. Publish a public web app

1. Create a GitHub repository for this project.
2. Add `app.py`, `requirements.txt`, and `travel_insurance_model.pkl` to the repository.
3. Open the official [Streamlit Community Cloud](https://share.streamlit.io/) site and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the app file to `app.py`.
5. Deploy. Streamlit will provide a shareable `*.streamlit.app` URL.

Only publish the model if you are comfortable making the model artifact public. Do not upload private customer data, passwords, API keys, or secrets. Model files are Python pickle artifacts: only load files you created or trust.

Official guides: [deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) and [manage dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Notes

- This app performs inference only; it does not retrain the model when a user submits the form.
- Predictions reflect the patterns learned from the training data, not certainty about an individual customer's future behaviour.
- Validate the model, class labels, probability calibration, and data handling before any real-world use.
