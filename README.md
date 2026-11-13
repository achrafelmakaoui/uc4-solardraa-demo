# ☀️ UC4 SolarDrâa: forecasting solar production

A small web app that shows our work on the **UC4 "Centrale solaire"** case study.  
M2 SDIA, *Advanced Artificial Intelligence*, ENSET Mohammedia, 2026–2027.

**Live app:** https://uc4-solardraa-demo.streamlit.app/

---

## The problem in simple words

SolarDrâa (a fictional company) runs two solar plants:

- **Site A**: 50 MW in the desert near Ouarzazate. We have 3 years of data, one row per hour.
- **Site B**: 20 MW on the Atlantic coast, with fog in the morning. We have only 2 months to learn from.

Every hour the company must say **how much electricity it will produce in each of the next 6 hours**.
If the forecast is wrong, it pays a penalty. Our job is to make that forecast better.

We also have:

- a **sky camera** that takes a small photo of the sky every hour, and
- **thermal photos** of the solar panels, taken by a drone, to find broken panels.

All the data are simulated for the course.

## How we measure "better"

We compare every model with a simple reference called **smart persistence**:
*"the sky in an hour will be like now; only the sun moves."*

> **gain = 1 − (our error / error of smart persistence)**

- **0** means as good as the reference.
- **0.30** means 30 % less error than the reference.
- **Negative** means worse than the reference.

We only count daytime hours, because at night production is always 0.  
We learn on the past and test on the future (2025), and we use the test data **only once**, at the end.

## What we did, in 4 steps

**Step 1: a simple neural network (MLP).**
It reads the last 24 hours of production and the weather forecast for the next 6 hours, then predicts the
next 6 hours. We found that the **cloud forecast** and the **recent history** are the most useful inputs, and
that training longer helped more than adding regularisation.

**Step 2: looking at images (CNN and auto-encoder).**

- A **CNN** looks at the sky photo and says how cloudy it is and whether the sun is hidden. It is twice as
  accurate as a simple rule. Adding what the CNN sees to the forecast **improves the 1-hour forecast**.
- An **auto-encoder** learns what a healthy panel looks like. When it sees a broken panel, it cannot rebuild
  it well, so the broken part lights up. This finds **88 % of cracked cells**.

**Step 3: a new site with little data (transfer learning).**
We reused the site A model on site B. Surprise: the site A model **used as it is** worked better in spring
than a version re-trained on site B's two winter months. The re-trained model "forgot" sunny days.

**Step 4: reading time as a sequence (RNN, LSTM, GRU).**
These networks read the past hour by hour. The **GRU** gave our best result on site B.
Looking further back (one week instead of 6 hours) did not help.

## Results on the test data

| Goal | Our result | Reached? |
|---|---|---|
| Gain at 1 hour, site A ≥ 0.12 | 0.133 | ✅ |
| Average gain, site A ≥ 0.30 | 0.290 | ❌ (very close) |
| Average gain, site B ≥ 0.27 | 0.300 | ✅ |
| Hidden-sun error ≤ 0.10 | 0.038 | ✅ |
| Broken-panel detection (AUC) ≥ 0.98 | 0.985 | ✅ |
| Cracked cells found ≥ 85 % | 88 % | ✅ |
| Model size ≤ 5 MB | ≤ 1.2 MB | ✅ |

**6 goals out of 7 reached.**

## What you can do in the app

| Page | What you see |
|---|---|
| **1 · Forecast explorer** | Pick a day: the real production compared with each model's forecast. |
| **2 · Sky camera** | A sky photo: what the camera measured vs what our models guessed. |
| **3 · Panel inspection** | A thermal photo of a panel, its rebuilt version, and where the defect is. |
| **4 · Test results** | All our scores, computed live by the saved models. |

Everything runs on our **real saved models**. Nothing is retrained and no number is typed into the code.

## Run it on your computer

```bash
git clone https://github.com/achrafelmakaoui/uc4-solardraa-demo.git
cd uc4-solardraa-demo
python -m pip install -r requirements.txt
python -m streamlit run demo_app.py
```

Then open http://localhost:8501. The first load takes about 30 seconds.

## What is in this repo

```
demo_app.py        the app
requirements.txt   the Python packages it needs
data_UC4/          the case-study data (simulated)
outputs/models/    our 8 trained models
```
