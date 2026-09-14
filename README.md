# Streaming Media IP Acquisition Modeling

## Predicting Community Momentum & Identifying Above-Expected Engagement

An end-to-end data analytics and machine learning project that uses MyAnimeList community data to identify anime/IPs with strong community momentum and unusually high engagement relative to their existing audience signals.

The project is framed around a realistic streaming-industry question:

> **How can a streaming platform use community data to identify anime/IPs that deserve further investigation for potential acquisition?**

---

## Business Report

[View the full Business & Technical Report](./Streaming_Media_IP_Acquisition_Business_Report-1(1).pdf)

## Project Overview

Streaming platforms have thousands of potential titles to evaluate. Acquisition teams need to balance audience demand, popularity, licensing availability, cost, competition, and long-term IP potential.

A title that is already extremely popular is relatively easy to identify.

The more interesting analytical question is:

> **Can data help identify titles showing stronger community engagement than we would expect from their existing audience and popularity?**

This project develops a data-driven screening framework using MyAnimeList data.

It does **not** attempt to automatically decide which anime a streaming platform should acquire.

Instead, it provides:

- A community-momentum classification model
- An interpretable machine learning approach
- Exploratory analysis of audience and engagement patterns
- An above-expected engagement model
- A ranked shortlist of titles that deserve further investigation

---

# Business Objective

The primary objective is to investigate whether measurable audience and popularity signals can identify anime with relatively strong community momentum.

The project addresses two related questions:

### Question 1 — Community Momentum

> Can existing audience size, popularity rank, and rating signals identify anime with relatively high community engagement?

### Question 2 — Above-Expected Engagement

> Which anime receive more community engagement than we would expect given their existing audience, popularity, and rating?

The second question is particularly useful for identifying potentially overlooked IPs.

---

# Business Use Case

A hypothetical streaming platform could use the workflow as an initial IP-screening layer:

```text
Large Anime Catalog
        │
        ▼
Audience & Popularity Data
        │
        ▼
Community Engagement Analysis
        │
        ▼
Machine Learning Screening
        │
        ▼
Above-Expected Engagement
        │
        ▼
Acquisition Shortlist
        │
        ▼
Business & Licensing Investigation
        │
        ▼
Final Acquisition Decision
```

The machine learning model is therefore a **decision-support tool**, not a replacement for business judgment.

---

# Dataset

The project uses the Kaggle dataset:

**Anime Dataset with Reviews - MyAnimeList**

Dataset:

https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews

The dataset contains MyAnimeList anime, review, and profile information collected around 2020.

## Files Used

```text
animes.csv
reviews.csv
```

`profiles.csv` was not required for this project.

---

# Dataset Scale

| Stage | Records |
|---|---:|
| Original anime records | 19,311 |
| Unique anime after cleaning | 16,216 |
| Original reviews | 192,112 |
| Unique reviews after cleaning | 130,519 |
| Anime used for ML | 2,266 |

Raw datasets are intentionally excluded from the GitHub repository because of their size.

---

# Project Workflow

The project is divided into four analytical notebooks:

```text
00_Data_Verification.ipynb
        ↓
01_Community_Features.ipynb
        ↓
02_EDA.ipynb
        ↓
03_ML_Modeling.ipynb
```

Each notebook has a specific responsibility.

---

# 1. Data Verification

### Notebook

`00_Data_Verification.ipynb`

The first stage focuses on making sure the data is suitable for analysis.

The following checks were performed:

- Dataset dimensions
- Column inspection
- Missing values
- Statistical distributions
- Duplicate anime IDs
- Duplicate review IDs
- Review-to-anime relationships
- Unusual review scores
- Data consistency

## Duplicate Handling

The anime dataset contained duplicate anime UIDs.

After deduplication:

```text
19,311 original anime records
            ↓
16,216 unique anime
```

The review dataset also contained duplicate review UID records.

After deduplication:

```text
192,112 original reviews
            ↓
130,519 unique reviews
```

All cleaned reviews matched an anime UID.

Therefore:

```text
Unmatched reviews = 0
```

---

# 2. Community Feature Engineering

### Notebook

`01_Community_Features.ipynb`

The next stage transforms raw review information into measurable community-engagement features.

Important features include:

| Feature | Meaning |
|---|---|
| `review_count` | Number of reviews received by an anime |
| `log_review_count` | Log-transformed review count |
| `members` | Number of MyAnimeList members |
| `popularity` | MyAnimeList popularity rank |
| `score` | Overall MyAnimeList score |
| `avg_review_score` | Average score given by reviewers |
| `score_std` | Variation in reviewer scores |
| `score_difference` | Difference between community review score and MAL score |
| `abs_score_difference` | Absolute difference between the two scores |

---

# Why Log Transform Review Count?

Review volume is extremely right-skewed.

Most anime have very few reviews, while a small number have hundreds or even more than 1,000.

A logarithmic transformation reduces the influence of extreme values:

```python
log_review_count = np.log1p(review_count)
```

This allows statistical models to work with the engagement distribution more effectively.

---

# 3. Exploratory Data Analysis

### Notebook

`02_EDA.ipynb`

The EDA stage investigated the relationship between:

- Audience size
- Popularity
- Ratings
- Review volume
- Review behavior
- Genres

---

# Key EDA Findings

## Review Count vs Members

The correlation between raw review count and member count was approximately:

```text
0.828
```

This indicates a strong positive relationship between audience size and community review activity.

In simple terms:

> Anime with larger existing audiences tend to generate more community engagement.

---

## Log Review Count vs Popularity

The correlation was approximately:

```text
-0.768
```

This negative relationship is important to interpret correctly.

MyAnimeList's popularity variable is a **rank**.

Therefore:

```text
Rank 1       = extremely popular
Rank 10,000  = less popular
```

A lower rank means greater popularity.

Therefore, the negative correlation is directionally expected.

---

## Log Review Count vs Members

The correlation was approximately:

```text
0.648
```

Again, this indicates that audience size is strongly related to community engagement.

---

# Engagement Is Not the Same as Sentiment

One of the useful findings from the EDA was that review volume and review sentiment are not interchangeable.

Some highly reviewed anime have very high ratings.

Others have substantially lower ratings.

This means:

> A title generating a lot of discussion is not necessarily a title that everyone likes.

For acquisition screening, this distinction matters.

A platform may want to investigate:

- Highly popular titles
- Highly rated titles
- Highly discussed titles
- Titles with unusually strong engagement relative to their audience

These are different signals.

---

# 4. Machine Learning

### Notebook

`03_ML_Modeling.ipynb`

The ML stage has two related components:

1. High-momentum classification
2. Above-expected engagement analysis

---

# High-Momentum Classification

The classification problem was defined as:

> Can existing audience and popularity signals identify anime with relatively high community engagement?

The modeling population was restricted to anime with at least:

```text
10 reviews
```

This resulted in:

```text
2,266 anime
```

---

# Defining High Momentum

Within the modeling population, the 75th percentile of review count was:

```text
50 reviews
```

Therefore:

```text
high_momentum = 1
```

when:

```text
review_count > 50
```

and:

```text
high_momentum = 0
```

when:

```text
review_count <= 50
```

This produced:

| Class | Anime |
|---|---:|
| Lower momentum | 1,709 |
| High momentum | 557 |
| Total | 2,266 |

The high-momentum class represents approximately:

```text
24.6%
```

of the modeling population.

---

# Why Not Use Review Count as a Predictor?

This is an important modeling decision.

The target itself is created from:

```text
review_count
```

Therefore, using:

```text
review_count
log_review_count
```

as predictors would directly leak information about the target.

That would make the model appear much better than it really is.

Instead, the final classification model uses:

```text
members
popularity
score
```

These are existing audience/popularity signals rather than direct measurements of the target.

---

# Final Classification Features

```text
members
popularity
score
```

Review-derived variables such as:

```text
avg_review_score
score_std
score_difference
abs_score_difference
```

were excluded from the final classification model.

The reason is to avoid making the business question too circular.

The cleaner question becomes:

> **Can existing audience, popularity, and rating signals identify strong community momentum without directly using review-derived engagement information?**

---

# Models Tested

Two machine learning algorithms were evaluated.

## Logistic Regression

An interpretable statistical classification model.

Advantages:

- Simple
- Fast
- Easy to interpret
- Useful for understanding feature relationships

## Random Forest

A tree-based ensemble model.

Advantages:

- Can model nonlinear relationships
- Can capture interactions between features
- Often performs well on tabular datasets

A simple majority-class model was also used as the baseline.

---

# Model Performance

All models were evaluated on a held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 0.753 | — | — | — | — |
| Logistic Regression | **0.848** | **0.705** | 0.661 | **0.682** | **0.904** |
| Random Forest | 0.813 | 0.596 | **0.750** | 0.664 | 0.877 |

---

# Final Model

## Logistic Regression

Logistic Regression was selected as the final classification model.

Its test-set performance was:

```text
Accuracy   = 0.848
Precision  = 0.705
Recall     = 0.661
F1 Score   = 0.682
ROC-AUC    = 0.904
```

The model substantially outperformed the majority-class baseline.

It also outperformed Random Forest in:

- Accuracy
- Precision
- F1
- ROC-AUC

Random Forest had higher recall, meaning it identified more of the high-momentum anime, but it also generated more false positives.

---

# What Does ROC-AUC 0.904 Mean?

ROC-AUC measures how well the model separates the two classes across different probability thresholds.

A value closer to:

```text
1.0
```

means stronger discrimination.

The final model achieved:

```text
ROC-AUC = 0.904
```

This indicates strong separation between the high-momentum and lower-momentum groups in the held-out test data.

---

# Confusion Matrix

The Logistic Regression confusion matrix was:

| | Predicted Low | Predicted High |
|---|---:|---:|
| **Actual Low** | 311 | 31 |
| **Actual High** | 38 | 74 |

Therefore:

- True Negatives = 311
- False Positives = 31
- False Negatives = 38
- True Positives = 74

The model correctly identified:

```text
74 / 112
```

of the actual high-momentum anime in the test set.

---

# Model Interpretability

The standardized Logistic Regression coefficients were:

| Feature | Coefficient |
|---|---:|
| Popularity | -3.195 |
| Score | -0.454 |
| Members | +0.358 |

Popularity had by far the strongest fitted relationship with the classification outcome.

Because MyAnimeList popularity is a rank, a negative coefficient is directionally consistent with:

```text
Lower popularity rank
        ↓
Greater popularity
        ↓
Higher likelihood of high community momentum
```

The coefficients describe relationships learned by the model.

They should **not** be interpreted as causal effects.

---

# Above-Expected Community Engagement

The classification model answers:

> Is this anime in the high-momentum group?

The project then asks a second question:

> Is this anime receiving more community engagement than we would expect from its audience signals?

For this analysis, a Linear Regression model was trained to predict:

```text
log_review_count
```

using:

```text
members
popularity
score
```

---

# Regression Performance

The model achieved:

| Metric | Result |
|---|---:|
| MAE | 0.413 |
| RMSE | 0.523 |
| R² | **0.628** |

An R² of 0.628 means the model explains approximately:

```text
62.8%
```

of the variation in log review count in the held-out test set.

---

# Momentum Residual

The project calculates:

```text
Momentum Residual
=
Actual Engagement
-
Expected Engagement
```

A positive residual means:

> The anime generated more community engagement than the model expected given its members, popularity, and score.

This creates a useful way of identifying potentially overlooked titles.

---

# Above-Expected Engagement Candidates

The strongest candidates included:

| Anime | Members | Popularity | Reviews |
|---|---:|---:|---:|
| Princess Tutu | 110,983 | 1,010 | 94 |
| Akatsuki no Yona | 441,757 | 174 | 239 |
| Kaze ga Tsuyoku Fuiteiru | 118,998 | 955 | 84 |
| Yuru Camp△ | 209,936 | 521 | 122 |
| Mawaru Penguindrum | 221,240 | 482 | 127 |
| Mo Dao Zu Shi | 76,350 | 1,418 | 60 |
| Lovely★Complex | 356,826 | 259 | 176 |
| Phantom: Requiem for the Phantom | 235,408 | 436 | 127 |
| Cross Game | 86,377 | 1,252 | 65 |

These are **not automatic acquisition recommendations**.

They are titles that the analytical framework flags for further investigation.

---

# Why This Is Interesting for a Streaming Platform

A conventional popularity-based approach might focus heavily on titles that already have very large audiences.

The residual analysis provides another perspective:

> **Which titles are generating more community engagement than their existing audience signals would suggest?**

This can help surface:

- Overlooked titles
- Niche titles with strong community interest
- Titles with unusually engaged audiences
- Potential catalog gaps
- IPs worth additional commercial investigation

The model therefore works as an initial **discovery and prioritization layer**.

---

# Example Business Workflow

Suppose the model flags an anime as an above-expected engagement candidate.

The streaming acquisition team could then investigate:

### Audience

- Who is engaging with the title?
- Which demographics are represented?
- Which regions show the strongest interest?

### Licensing

- Are streaming rights available?
- Which territories are available?
- What is the estimated licensing cost?

### Competition

- Which competing platforms already carry the title?
- Is the IP exclusive anywhere?

### Commercial Potential

- Could the title attract new subscribers?
- Does it complement the existing catalog?
- Does it have sequel, franchise, merchandising, or adaptation potential?

The machine learning model only identifies the title as a **candidate for investigation**.

The final business decision requires all of these additional factors.

---

# Key Business Insight

The most valuable output of the project is not simply:

> "Which anime are popular?"

Instead, it asks:

> **"Which anime demonstrate stronger community engagement than we would expect from their existing audience and popularity signals?"**

This produces a more interesting acquisition-screening strategy than simply ranking titles by popularity.

---

# Important Limitations

## 1. Cross-Sectional Dataset

The available MyAnimeList data is a snapshot rather than a detailed longitudinal dataset.

Therefore, the project cannot determine whether community momentum occurred **before** future success.

It measures relationships within the available data.

---

## 2. Community Engagement ≠ Streaming Success

Review activity on MyAnimeList is only a proxy for community engagement.

It is not equivalent to:

- Streaming hours
- Watch completion
- Subscriber growth
- Revenue
- Profitability

---

## 3. MyAnimeList Audience ≠ Entire Streaming Audience

MyAnimeList users are not necessarily representative of every streaming platform's audience.

The results should therefore not be generalized to all viewers without additional validation.

---

## 4. Missing Business Variables

The current model does not include:

- Licensing availability
- Licensing cost
- Regional rights
- Demographics
- Streaming watch time
- Completion rates
- Subscriber impact
- Competition
- Marketing cost
- Franchise economics

These would be essential for a production acquisition system.

---

## 5. No Causal Claims

The project identifies statistical associations.

It does not establish that:

```text
Popularity → Community Engagement → Streaming Success
```

is a causal chain.

The model should therefore be treated as a **screening signal**, not a causal forecasting system.

---

# Future Improvements

A production-level version of this project could incorporate additional data.

### Longitudinal Data

Collect community metrics over time to determine whether momentum precedes later growth.

### Streaming Data

Include:

- Watch hours
- Completion rate
- Repeat viewing
- Subscriber acquisition
- Subscriber retention

### Regional Analysis

Measure audience interest across:

- Countries
- Languages
- Regions

### Commercial Data

Add:

- Licensing cost
- Rights availability
- Contract duration
- Exclusivity
- Market competition

### Audience Demographics

Analyze:

- Age groups
- Viewer segments
- Regional audience profiles

### Additional Community Signals

Potential future sources could include:

- Search trends
- Social-media engagement
- Discussion activity
- External ratings
- Community growth over time

---

# Project Structure

```text
StreamingMediaIPModel/
│
├── data/
│   ├── raw/
│   │   ├── animes.csv
│   │   └── reviews.csv
│   │
│   ├── anime_features.csv
│   └── genre_summary.csv
│
├── notebooks/
│   ├── 00_Data_Verification.ipynb
│   ├── 01_Community_Features.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_ML_Modeling.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── src/
│
├── .gitignore
└── .venv/
```

---

# Notebook Guide

## `00_Data_Verification.ipynb`

Responsible for:

- Data loading
- Schema inspection
- Missing-value analysis
- Duplicate detection
- Distribution checks
- Referential integrity

---

## `01_Community_Features.ipynb`

Responsible for:

- Review aggregation
- Community engagement features
- Log transformation
- Score comparisons
- Feature dataset creation

---

## `02_EDA.ipynb`

Responsible for:

- Review-volume analysis
- Audience analysis
- Rating analysis
- Correlation analysis
- Genre analysis
- Community behavior exploration

---

## `03_ML_Modeling.ipynb`

Responsible for:

- Target definition
- Train/test splitting
- Baseline model
- Logistic Regression
- Random Forest
- Model evaluation
- Confusion matrix
- ROC curve
- Regression modeling
- Momentum residual analysis
- Acquisition-screening shortlist

---

# Technology Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Model Persistence

- Joblib

### Development

- Jupyter Notebook
- PyCharm

### Version Control

- Git
- GitHub

---

# Project Outputs

The project generates processed datasets:

```text
data/anime_features.csv
data/genre_summary.csv
```

Machine learning model:

```text
outputs/models/logistic_momentum_model.pkl
```

Model evaluation:

```text
outputs/tables/model_results.csv
outputs/tables/final_model_summary.csv
```

Business screening outputs:

```text
outputs/tables/hidden_momentum_candidates.csv
outputs/tables/acquisition_candidates.csv
```

---

# Reproducing the Project

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd StreamingMediaIPModel
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
```

## 5. Add the dataset

Download the MyAnimeList dataset and place:

```text
animes.csv
reviews.csv
```

inside:

```text
data/raw/
```

The raw dataset is intentionally not included in the repository.

## 6. Run the notebooks

Run the notebooks in order:

```text
00_Data_Verification.ipynb
        ↓
01_Community_Features.ipynb
        ↓
02_EDA.ipynb
        ↓
03_ML_Modeling.ipynb
```

---

# Final Results

### Classification

```text
Final Model: Logistic Regression

Accuracy:   0.848
Precision:  0.705
Recall:     0.661
F1 Score:   0.682
ROC-AUC:    0.904
```

### Above-Expected Engagement Regression

```text
MAE:  0.413
RMSE: 0.523
R²:   0.628
```

---

# Final Takeaway

This project demonstrates how community data can be transformed into a practical business-screening framework.

Rather than simply asking:

> **"What anime are already popular?"**

the project asks:

> **"Which anime show unusually strong community engagement relative to their observable audience signals?"**

The classification model provides a high-momentum screening signal, while the regression residual analysis identifies titles that appear to be generating more engagement than expected.

Together, these approaches can help a hypothetical streaming platform move from a large catalog to a smaller group of IPs that deserve deeper investigation.

However, the final decision should always combine the analytical signal with:

```text
Data
+
Licensing
+
Audience
+
Competition
+
Cost
+
Regional Demand
+
Business Strategy
```

The model does not say:

> **"Acquire this anime."**

It says:

> **"This title shows a pattern worth investigating."**

---

# Disclaimer

This is an educational and portfolio data science project.

The model is based on MyAnimeList community data and should not be interpreted as a production recommendation system, investment advice, or a guarantee of future streaming performance.

The results represent statistical relationships within the available dataset. Real-world acquisition decisions would require current, longitudinal, commercially relevant, and independently validated data.

---

## Author

**Priyansh Bobade**

Data Analytics & Machine Learning Project

```text
Python • Pandas • NumPy • Matplotlib • Seaborn • Scikit-learn
```
