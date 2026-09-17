# CV RAG Job Assistant (Enhanced)

A practical Streamlit application that helps you improve your CV and job applications using Retrieval-Augmented Generation (RAG) techniques.

## Features

| Feature | Description |
|---------|-------------|
| **ATS Rating (1-10)** | Heuristic score of how well your CV is likely to pass Applicant Tracking Systems |
| **CV Downfalls & Advice** | Highlights weaknesses and gives concrete improvement tips |
| **Skills Extraction** | Automatically pulls technical & soft skills from your CV |
| **Job Match Score** | Paste any job description → get a 1-10 similarity score + missing keywords |
| **Relevant CV Snippets** | RAG-style retrieval of the most relevant sentences from your CV |
| **Cover Letters (3 styles)** | Professional, Enthusiastic, or Concise – tailored to the job |
| **LinkedIn About Generator** | Creates a ready-to-paste professional summary for LinkedIn |
| **Suggested Job Titles** | Recommends roles that match your skills and experience level |
| **Live Job Search** | Best-effort online search + direct links to LinkedIn / Indeed / Glassdoor |
| **Export Report** | Download full analysis as a Markdown file |

## How to Run

```bash
# 1. Install dependencies (recommended inside a virtual environment)
pip install -r requirements.txt

# 2. Launch
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## How to Use

1. **Upload** your CV (PDF or DOCX) in the sidebar.
2. Explore the tabs:
   - **CV Analysis & ATS** → score, strengths, weaknesses, tips + download report
   - **Match vs Job** → paste a real job description for a match rating
   - **Cover Letter** → generate a personalized letter in 3 different styles
   - **LinkedIn About** → create a polished LinkedIn summary
   - **Suggested Jobs** → list of roles that fit your profile
   - **Search Live Jobs** → try an online search or use the quick links

## Technical Notes

- **RAG**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) when available, otherwise falls back to TF-IDF + cosine similarity.
- **ATS scoring** is heuristic. Useful guide, not a guarantee.
- Online job search is best-effort; many job boards block scrapers. Direct search links are always provided.
- Experience is calculated from actual date ranges (not crude year counting).

## Requirements

See `requirements.txt`. Python 3.9+ recommended.

Enjoy improving your applications!
