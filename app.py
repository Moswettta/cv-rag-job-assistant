"""
CV RAG Job Assistant  (Enhanced)
A Streamlit app that uses RAG (embeddings + TF-IDF) to:
- Analyze your CV for ATS readiness, strengths & weaknesses
- Score your CV 1-10 and rate it against any job
- Generate tailored application / cover letters (multiple styles)
- Generate LinkedIn "About" section
- Suggest related jobs + search live openings
- Export full analysis report
"""

import streamlit as st
import pdfplumber
from docx import Document
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO

# Optional better embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    USE_EMBEDDINGS = True
except Exception:
    USE_EMBEDDINGS = False
    EMBED_MODEL = None

st.set_page_config(
    page_title="CV RAG Job Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Utility functions
# ---------------------------

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> list:
    """Heuristic skill extraction from common tech & soft skill keywords."""
    common_skills = [
        "python", "java", "javascript", "typescript", "react", "node", "sql", "nosql",
        "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git", "ci/cd",
        "machine learning", "deep learning", "nlp", "data analysis", "pandas", "numpy",
        "tensorflow", "pytorch", "scikit-learn", "spark", "hadoop", "tableau", "power bi",
        "excel", "r", "c++", "c#", "go", "rust", "php", "ruby", "swift", "kotlin",
        "html", "css", "angular", "vue", "django", "flask", "fastapi", "spring",
        "rest api", "graphql", "microservices", "agile", "scrum", "jira",
        "leadership", "communication", "project management", "teamwork", "problem solving",
        "analytical", "presentation", "negotiation", "customer service", "sales",
        "marketing", "seo", "content writing", "copywriting", "accounting", "finance",
        "hr", "recruiting", "teaching", "research", "writing", "editing",
        "cybersecurity", "networking", "devops", "cloud", "blockchain", "iot",
        "ui/ux", "figma", "adobe", "photoshop", "illustrator", "video editing",
    ]
    text_lower = text.lower()
    found = []
    for skill in common_skills:
        if skill in text_lower:
            found.append(skill.title() if len(skill) > 3 else skill.upper())
    skills_section = re.search(
        r"(?:skills|technical skills|core competencies)[:\s]*(.+?)(?:\n\n|\Z)", text, re.I | re.S
    )
    if skills_section:
        extra = re.findall(r"[A-Za-z+#./ ]{2,30}", skills_section.group(1))
        for e in extra:
            e = e.strip()
            if len(e) > 2 and e.lower() not in [s.lower() for s in found]:
                found.append(e)
    return sorted(list(set(found)))[:40]


def extract_experience_years(text: str) -> float:
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
        r"experience[:\s]*(\d+)\+?\s*(?:years?|yrs?)",
    ]
    years = []
    for p in patterns:
        matches = re.findall(p, text, re.I)
        years.extend([int(m) for m in matches])
    if years:
        return max(years)
    job_markers = len(re.findall(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}", text, re.I))
    return max(1.0, job_markers * 1.5) if job_markers else 0


def ats_score(text: str, skills: list) -> dict:
    """Heuristic ATS score (1-10) + breakdown."""
    score = 5.0
    reasons = []

    words = len(text.split())
    if 300 <= words <= 900:
        score += 1.5
        reasons.append("✅ Good length (300-900 words)")
    elif words < 200:
        score -= 1.5
        reasons.append("❌ Too short – add more detail")
    else:
        score -= 0.5
        reasons.append("⚠️ Somewhat long – consider tightening")

    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text):
        score += 0.5
        reasons.append("✅ Email present")
    else:
        score -= 1
        reasons.append("❌ Missing email")

    if re.search(r"(\+?\d[\d\s\-]{7,})", text) or "linkedin.com" in text.lower():
        score += 0.5
        reasons.append("✅ Phone or LinkedIn present")

    sections = ["experience", "education", "skills", "projects", "summary", "objective"]
    found_sections = sum(1 for s in sections if s in text.lower())
    score += min(found_sections * 0.4, 1.5)
    if found_sections >= 3:
        reasons.append(f"✅ Clear sections detected ({found_sections})")
    else:
        reasons.append("⚠️ Add clearer section headers (Experience, Education, Skills...)")

    if len(skills) >= 8:
        score += 1.0
        reasons.append(f"✅ Good number of skills detected ({len(skills)})")
    elif len(skills) >= 4:
        score += 0.5
        reasons.append(f"⚠️ Moderate skills listed ({len(skills)})")
    else:
        score -= 0.5
        reasons.append("❌ Few skills detected – expand Skills section")

    action_verbs = [
        "developed", "led", "managed", "implemented", "designed", "built",
        "created", "improved", "increased", "reduced", "achieved", "collaborated"
    ]
    verb_count = sum(1 for v in action_verbs if v in text.lower())
    if verb_count >= 5:
        score += 0.8
        reasons.append("✅ Strong action verbs used")
    else:
        reasons.append("⚠️ Use more action verbs (led, developed, implemented...)")

    if text.count("|") > 10 or text.count("\t") > 20:
        score -= 0.7
        reasons.append("⚠️ Possible tables/complex formatting – ATS may struggle")

    if re.search(r"\d+%|\$\d+|\d+\s*(?:users|clients|projects|team)", text, re.I):
        score += 0.7
        reasons.append("✅ Quantifiable results present")
    else:
        reasons.append("⚠️ Add numbers & measurable impact")

    score = max(1.0, min(10.0, round(score, 1)))
    return {"score": score, "reasons": reasons}


def similarity_score(cv_text: str, job_text: str) -> float:
    if USE_EMBEDDINGS and EMBED_MODEL:
        emb1 = EMBED_MODEL.encode([cv_text])
        emb2 = EMBED_MODEL.encode([job_text])
        return float(cosine_similarity(emb1, emb2)[0][0])
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf = vectorizer.fit_transform([cv_text, job_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


def missing_keywords(cv_text: str, job_text: str, top_n: int = 15) -> list:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=100, ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([job_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        top_idx = scores.argsort()[::-1][:40]
        job_keywords = [feature_names[i] for i in top_idx if scores[i] > 0.05]
        cv_lower = cv_text.lower()
        missing = [kw for kw in job_keywords if kw.lower() not in cv_lower]
        return missing[:top_n]
    except Exception:
        return []


def generate_cover_letter(
    cv_text: str,
    job_title: str,
    company: str,
    job_desc: str,
    skills: list,
    style: str = "Professional",
) -> str:
    """Generate cover letter in different styles."""
    name_match = re.search(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", cv_text)
    name = name_match.group(1) if name_match else "Applicant"
    relevant_skills = ", ".join(skills[:8]) if skills else "relevant technical and soft skills"

    job_words = set(re.findall(r"\b\w{4,}\b", job_desc.lower()))
    sentences = re.split(r"[.!?]\s+", cv_text)
    scored = []
    for s in sentences:
        if len(s) < 30:
            continue
        overlap = len(set(re.findall(r"\b\w{4,}\b", s.lower())) & job_words)
        if overlap > 1:
            scored.append((overlap, s.strip()))
    scored.sort(reverse=True)
    highlights = " ".join([s[1] for s in scored[:3]]) if scored else "my professional experience and achievements."

    company_part = f" at {company}" if company else ""
    title_part = job_title or "the position"

    if style == "Professional":
        letter = f"""Dear Hiring Manager{(' at ' + company) if company else ''},

I am writing to express my strong interest in the {title_part} role{company_part}. With a background that includes {relevant_skills}, I am confident that my experience aligns well with the requirements of this role.

{highlights}

In my previous roles I have consistently delivered results through a combination of technical expertise, collaboration, and a focus on measurable outcomes. I am particularly drawn to this opportunity because of the chance to contribute to {company or 'your team'}'s goals and grow further in this domain.

I would welcome the opportunity to discuss how my skills and experience can benefit your team. Thank you for considering my application. I look forward to the possibility of speaking with you soon.

Sincerely,
{name}
"""
    elif style == "Enthusiastic":
        letter = f"""Dear Hiring Team{(' at ' + company) if company else ''},

I am excited to apply for the {title_part} position{company_part}! Your work in this area truly resonates with me, and I believe my skills in {relevant_skills} make me a strong fit.

{highlights}

I thrive in collaborative environments and love turning ideas into measurable results. Joining {company or 'your team'} would be an amazing opportunity to contribute my energy and expertise while continuing to grow.

Thank you for your time and consideration — I would love the chance to chat further!

Best regards,
{name}
"""
    else:  # Concise
        letter = f"""Dear Hiring Manager,

I am applying for the {title_part} role{company_part}. My background includes {relevant_skills}.

Key highlights:
{highlights}

I am confident I can add value quickly and would welcome the opportunity to discuss this further.

Sincerely,
{name}
"""
    return letter.strip()


def generate_linkedin_about(cv_text: str, skills: list, years: float) -> str:
    """Generate a LinkedIn About / Summary section."""
    name_match = re.search(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", cv_text)
    name = name_match.group(1) if name_match else "Professional"
    skill_str = ", ".join(skills[:10]) if skills else "various technical and soft skills"
    level = "Senior" if years >= 5 else ("Experienced" if years >= 2 else "Emerging")

    # Pull a couple of strong sentences
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", cv_text) if 40 < len(s.strip()) < 180]
    highlights = " ".join(sentences[:2]) if sentences else "I focus on delivering high-impact results."

    about = f"""{level} professional with {years:.0f}+ years of experience specializing in {skill_str}.

{highlights}

I am passionate about continuous learning, collaboration, and turning complex problems into elegant, scalable solutions. Always open to new opportunities where I can contribute and grow.

Core strengths: {skill_str}.
"""
    return about.strip()


def suggest_job_titles(skills: list, years: float) -> list:
    skill_set = set(s.lower() for s in skills)
    suggestions = []

    def has(*keys):
        return any(k in skill_set for k in keys)

    level = "Senior " if years >= 5 else ("Mid-level " if years >= 2 else "Junior ")

    if has("python", "machine learning", "tensorflow", "pytorch", "nlp", "data analysis"):
        suggestions.extend([f"{level}Machine Learning Engineer", f"{level}Data Scientist", "AI Engineer"])
    if has("python", "sql", "pandas", "tableau", "power bi", "excel"):
        suggestions.extend([f"{level}Data Analyst", "Business Intelligence Analyst"])
    if has("react", "javascript", "typescript", "node", "html", "css"):
        suggestions.extend([f"{level}Frontend Developer", f"{level}Full Stack Developer"])
    if has("java", "spring", "microservices", "sql"):
        suggestions.extend([f"{level}Backend Developer", f"{level}Java Developer"])
    if has("aws", "azure", "docker", "kubernetes", "devops", "ci/cd", "linux"):
        suggestions.extend([f"{level}DevOps Engineer", "Cloud Engineer", "Site Reliability Engineer"])
    if has("cybersecurity", "networking"):
        suggestions.extend(["Cybersecurity Analyst", "Security Engineer"])
    if has("project management", "agile", "scrum", "jira", "leadership"):
        suggestions.extend(["Project Manager", "Scrum Master"])
    if has("marketing", "seo", "content writing"):
        suggestions.extend(["Digital Marketing Specialist", "Content Marketing Manager"])
    if has("sales", "customer service", "negotiation"):
        suggestions.extend(["Sales Representative", "Account Manager"])
    if has("ui/ux", "figma", "adobe"):
        suggestions.extend(["UI/UX Designer", "Product Designer"])

    if not suggestions:
        suggestions = [f"{level}Software Engineer", f"{level}Software Developer", "Technical Specialist", "IT Consultant"]

    return list(dict.fromkeys(suggestions))[:12]


def search_jobs_online(query: str, location: str = "remote", num: int = 8) -> list:
    results = []
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": f"{query} jobs {location}"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.post(url, data=params, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select(".result__a")[:num]:
            title = a.get_text(strip=True)
            link = a.get("href", "")
            snippet_el = a.find_parent("div", class_="result")
            snippet = ""
            if snippet_el:
                sn = snippet_el.select_one(".result__snippet")
                if sn:
                    snippet = sn.get_text(strip=True)
            if title:
                results.append({"title": title, "link": link, "snippet": snippet})
    except Exception:
        pass
    return results


def build_analysis_report(cv_text, skills, ats, years, suggestions) -> str:
    """Create a downloadable Markdown report."""
    report = f"""# CV Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overall ATS Score: {ats['score']} / 10

### Score Breakdown
"""
    for r in ats["reasons"]:
        report += f"- {r}\n"

    report += f"""
## Detected Skills ({len(skills)})
{', '.join(skills) if skills else 'None clearly detected'}

## Estimated Experience
{years:.0f}+ years

## Suggested Job Titles
"""
    for i, t in enumerate(suggestions, 1):
        report += f"{i}. {t}\n"

    report += """
## Quick Improvement Tips
- Use standard section headers: Summary, Experience, Education, Skills
- Start bullets with strong action verbs + numbers
- Mirror keywords from target job descriptions
- Keep formatting simple (ATS-friendly)
- Aim for 1 page (junior) or 2 pages (senior)
- Include LinkedIn / portfolio / GitHub if relevant

## CV Preview (first 1200 characters)
"""
    report += cv_text[:1200] + ("..." if len(cv_text) > 1200 else "")
    return report


# ---------------------------
# Session state
# ---------------------------
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""
if "skills" not in st.session_state:
    st.session_state.skills = []
if "ats" not in st.session_state:
    st.session_state.ats = None
if "years" not in st.session_state:
    st.session_state.years = 0

# ---------------------------
# UI
# ---------------------------
st.title("📄 CV RAG Job Assistant")
st.markdown(
    "Upload your CV → get **ATS score**, **strengths & weaknesses**, "
    "**job match rating**, **tailored cover letters**, **LinkedIn About**, "
    "and **related job suggestions**."
)

with st.sidebar:
    st.header("1. Upload your CV")
    uploaded = st.file_uploader("PDF or DOCX", type=["pdf", "docx"])
    if uploaded:
        with st.spinner("Extracting text..."):
            if uploaded.name.lower().endswith(".pdf"):
                raw = extract_text_from_pdf(uploaded)
            else:
                raw = extract_text_from_docx(uploaded)
            st.session_state.cv_text = clean_text(raw)
            st.session_state.skills = extract_skills(st.session_state.cv_text)
            st.session_state.years = extract_experience_years(st.session_state.cv_text)
            st.session_state.ats = ats_score(st.session_state.cv_text, st.session_state.skills)
        st.success("CV loaded successfully!")

    st.markdown("---")
    st.caption("RAG engine: " + ("Embeddings (sentence-transformers)" if USE_EMBEDDINGS else "TF-IDF fallback"))
    st.caption(f"Ready • {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Main content
if not st.session_state.cv_text:
    st.info("👈 Please upload your CV (PDF or Word) in the sidebar to get started.")
    st.markdown(
        """
        ### What this tool does
        | Feature | Description |
        |---------|-------------|
        | **ATS Rating** | Scores your CV 1-10 for Applicant Tracking Systems |
        | **Downfalls & Advice** | Points out weaknesses and how to improve |
        | **Job Match** | Rate your CV against any job description |
        | **Cover Letter** | Generate tailored letters in 3 styles |
        | **LinkedIn About** | Create a ready-to-use LinkedIn summary |
        | **Job Suggestions** | Roles that match your skills + experience |
        | **Online Job Search** | Best-effort search + direct job-board links |
        | **Export Report** | Download full analysis as Markdown |
        """
    )
else:
    cv = st.session_state.cv_text
    skills = st.session_state.skills
    ats = st.session_state.ats
    years = st.session_state.years
    suggestions = suggest_job_titles(skills, years)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 CV Analysis & ATS",
        "🎯 Match vs Job",
        "✉️ Cover Letter",
        "🔗 LinkedIn About",
        "💼 Suggested Jobs",
        "🔍 Search Live Jobs"
    ])

    # ---------- Tab 1: Analysis ----------
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ATS Score", f"{ats['score']} / 10")
        with col2:
            st.metric("Detected Skills", len(skills))
        with col3:
            st.metric("Est. Experience", f"{years:.0f}+ years" if years else "Not clear")

        st.subheader("Score Breakdown & Advice")
        for r in ats["reasons"]:
            st.write(r)

        st.subheader("Detected Skills")
        if skills:
            st.write(", ".join(skills))
        else:
            st.warning("Few skills detected. Add a clear Skills section with keywords.")

        st.subheader("⚠️ Potential Downfalls")
        downs = [r for r in ats["reasons"] if r.startswith("❌") or r.startswith("⚠️")]
        if downs:
            for d in downs:
                st.write(d)
        else:
            st.success("No major red flags detected by the heuristics!")

        st.subheader("💡 Quick Improvement Tips")
        st.markdown(
            """
            - Use standard section headers: **Summary, Experience, Education, Skills**
            - Start bullets with strong action verbs + numbers (e.g. “Increased conversion by 27%”)
            - Mirror keywords from the job description you are applying to
            - Keep formatting simple (no tables, text boxes, or heavy graphics)
            - Aim for 1 page (junior) or 2 pages (senior)
            - Include LinkedIn / portfolio / GitHub if relevant
            """
        )

        st.subheader("CV Preview (first 1500 chars)")
        st.text_area("Extracted text", cv[:1500] + ("..." if len(cv) > 1500 else ""), height=200)

        # Export full report
        report = build_analysis_report(cv, skills, ats, years, suggestions)
        st.download_button(
            "📥 Download Full Analysis Report (.md)",
            report,
            file_name="cv_analysis_report.md",
            mime="text/markdown",
        )

    # ---------- Tab 2: Match vs Job ----------
    with tab2:
        st.subheader("Rate your CV against a specific job")
        job_title = st.text_input("Job Title", placeholder="e.g. Senior Data Scientist", key="match_title")
        company = st.text_input("Company (optional)", placeholder="e.g. Acme Corp", key="match_company")
        job_desc = st.text_area(
            "Paste the full Job Description here",
            height=220,
            placeholder="Paste the requirements, responsibilities, qualifications...",
            key="match_desc",
        )

        if st.button("🔍 Analyze Match", type="primary") and job_desc.strip():
            with st.spinner("Computing similarity (RAG)..."):
                sim = similarity_score(cv, job_desc)
                match_score = round(sim * 10, 1)
                missing = missing_keywords(cv, job_desc)

            st.metric("Match Score (CV ↔ Job)", f"{match_score} / 10")
            st.progress(min(sim, 1.0))

            if match_score >= 7.5:
                st.success("Strong match! Your CV already covers most key points.")
            elif match_score >= 5:
                st.warning("Moderate match. Tailor your CV and cover letter to close the gap.")
            else:
                st.error("Low match. Consider adding missing keywords or targeting a better-fitting role.")

            st.subheader("Missing / Under-emphasized Keywords")
            if missing:
                st.write(", ".join(missing))
                st.caption("Try to weave these naturally into your Experience or Skills sections.")
            else:
                st.write("No major missing keywords detected.")

            st.subheader("Most Relevant Parts of Your CV (RAG)")
            job_words = set(re.findall(r"\b\w{4,}\b", job_desc.lower()))
            sentences = [s.strip() for s in re.split(r"[.!?]\s+", cv) if len(s.strip()) > 40]
            scored = []
            for s in sentences:
                overlap = len(set(re.findall(r"\b\w{4,}\b", s.lower())) & job_words)
                scored.append((overlap, s))
            scored.sort(reverse=True)
            for _, s in scored[:5]:
                st.write(f"• {s}")

    # ---------- Tab 3: Cover Letter ----------
    with tab3:
        st.subheader("Generate Application / Cover Letter")
        cl_job = st.text_input("Job Title for letter", key="cl_job")
        cl_company = st.text_input("Company name", key="cl_comp")
        cl_desc = st.text_area("Job description (for better tailoring)", height=150, key="cl_desc")
        style = st.selectbox("Letter style", ["Professional", "Enthusiastic", "Concise"])

        if st.button("✍️ Generate Letter", type="primary"):
            letter = generate_cover_letter(
                cv, cl_job, cl_company, cl_desc or "the role", skills, style=style
            )
            st.text_area("Your tailored cover letter", letter, height=420)
            st.download_button("Download as .txt", letter, file_name="cover_letter.txt")

    # ---------- Tab 4: LinkedIn About ----------
    with tab4:
        st.subheader("Generate LinkedIn About / Summary")
        st.caption("A ready-to-paste professional summary for your LinkedIn profile.")
        if st.button("🔗 Generate LinkedIn About", type="primary"):
            about = generate_linkedin_about(cv, skills, years)
            st.text_area("LinkedIn About section", about, height=280)
            st.download_button("Download as .txt", about, file_name="linkedin_about.txt")

    # ---------- Tab 5: Suggested Jobs ----------
    with tab5:
        st.subheader("Roles you can apply for (based on your CV)")
        for i, title in enumerate(suggestions, 1):
            st.write(f"{i}. **{title}**")

        st.markdown("---")
        st.subheader("How to use these suggestions")
        st.markdown(
            """
            1. Search these titles on LinkedIn, Indeed, Glassdoor, Wellfound, etc.
            2. For each interesting posting, paste the job description into the **Match vs Job** tab.
            3. Improve your CV with the missing keywords, then generate a fresh cover letter.
            """
        )

    # ---------- Tab 6: Live Search ----------
    with tab6:
        st.subheader("Search available jobs on the internet")
        default_query = suggestions[0] if suggestions else "software engineer"
        q = st.text_input("Search query", value=default_query, key="search_q")
        loc = st.text_input("Location preference", value="remote", key="search_loc")
        if st.button("🔎 Search Jobs"):
            with st.spinner("Searching (DuckDuckGo)..."):
                jobs = search_jobs_online(q, loc)
            if jobs:
                for j in jobs:
                    st.markdown(f"**[{j['title']}]({j['link']})**")
                    if j["snippet"]:
                        st.caption(j["snippet"][:220])
                    st.write("---")
            else:
                st.info(
                    "No direct results returned (search engines often block automated queries). "
                    "Use the suggested titles above and search manually on LinkedIn / Indeed / Glassdoor."
                )
            st.markdown(
                f"""
                **Quick links you can try:**
                - [LinkedIn jobs for “{q}”](https://www.linkedin.com/jobs/search/?keywords={q.replace(' ', '%20')})
                - [Indeed “{q}”](https://www.indeed.com/jobs?q={q.replace(' ', '+')}&l={loc})
                - [Glassdoor “{q}”](https://www.glassdoor.com/Job/jobs.htm?sc.keyword={q.replace(' ', '+')})
                """
            )

st.markdown("---")
st.caption(
    "Built with Streamlit + RAG (sentence-transformers / TF-IDF). "
    "Heuristic ATS scoring – always review the output manually before applying."
)
