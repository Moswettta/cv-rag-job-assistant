"""
CV RAG Job Assistant  (Enhanced)
A Streamlit app that uses RAG (embeddings + TF-IDF) to:
- Analyze your CV for ATS readiness, strengths & weaknesses
- Score your CV 1-10 and rate it against any job
- Generate tailored application / cover letters (multiple styles)
- Generate LinkedIn "About" section
- Suggest related jobs + search live openings across ALL major careers
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
    common_skills = [
        "python", "java", "javascript", "typescript", "react", "node", "sql", "nosql",
        "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git", "ci/cd",
        "machine learning", "deep learning", "nlp", "data analysis", "pandas", "numpy",
        "tensorflow", "pytorch", "scikit-learn", "spark", "hadoop", "tableau", "power bi",
        "excel", "r", "c++", "c#", "go", "rust", "php", "ruby", "swift", "kotlin",
        "html", "css", "angular", "vue", "django", "flask", "fastapi", "spring",
        "rest api", "graphql", "microservices", "agile", "scrum", "jira",
        "cybersecurity", "networking", "devops", "cloud", "blockchain", "iot",
        "ui/ux", "figma", "adobe", "photoshop", "illustrator", "video editing",
        "oracle", "windows", "tcp/ip", "dns", "dhcp", "http", "lan/wan", "vmware", "virtualbox",
        "end-user", "incident", "troubleshooting", "mysql", "html5", "css3",
        "leadership", "communication", "project management", "teamwork", "problem solving",
        "analytical", "presentation", "negotiation", "customer service", "sales",
        "marketing", "seo", "content writing", "copywriting", "accounting", "finance",
        "hr", "recruiting", "teaching", "research", "writing", "editing",
        "time management", "multitasking", "attention to detail", "hygiene",
        "organization", "planning", "reporting", "data entry", "administration",
        "barista", "espresso", "coffee", "tea", "specialty drinks", "beverage",
        "coffee machine", "grinder", "blender", "latte", "cappuccino", "cafe",
        "hospitality", "food service", "waiter", "waitress", "server", "cashier",
        "pos", "point of sale", "menu", "order management", "customer experience",
        "cleanliness", "sanitation", "food safety", "kitchen", "restaurant", "chef", "cook",
        "nursing", "nurse", "patient care", "clinical", "medical", "pharmacy", "pharmacist",
        "first aid", "cpr", "vital signs", "healthcare", "hospital", "clinic", "lab",
        "phlebotomy", "radiology", "physiotherapy", "counseling", "mental health",
        "teaching", "teacher", "tutor", "curriculum", "lesson planning", "classroom",
        "student", "education", "training", "facilitation", "mentoring", "coaching",
        "accounting", "bookkeeping", "financial reporting", "audit", "taxation", "budgeting",
        "quickbooks", "sage", "banking", "teller", "credit", "loan", "investment",
        "reconciliation", "accounts payable", "accounts receivable", "payroll",
        "sales", "retail", "merchandising", "inventory", "stock", "cashier", "pos",
        "crm", "lead generation", "cold calling", "negotiation", "branding", "social media",
        "digital marketing", "seo", "sem", "content marketing", "email marketing",
        "administration", "office management", "receptionist", "front desk", "secretarial",
        "filing", "scheduling", "calendar", "ms office", "microsoft office", "data entry",
        "typing", "transcription", "minutes", "correspondence",
        "logistics", "supply chain", "warehouse", "inventory", "procurement", "shipping",
        "freight", "driver", "driving", "delivery", "fleet", "transport",
        "graphic design", "design", "photoshop", "illustrator", "indesign", "canva",
        "photography", "videography", "editing", "content creation", "social media management",
        "writing", "copywriting", "journalism", "broadcasting",
        "mechanical", "electrical", "civil", "engineering", "autocad", "solidworks",
        "maintenance", "technician", "welding", "plumbing", "construction",
        "security", "guard", "surveillance", "cctv", "safety", "hse", "fire safety",
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
    for p in patterns:
        matches = re.findall(p, text, re.I)
        if matches:
            return float(max(int(m) for m in matches))
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    range_pat = re.compile(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})"
        r"\s*[–\-—to]+\s*"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})",
        re.I,
    )
    total_months = 0
    for m1, y1, m2, y2 in range_pat.findall(text):
        try:
            start = int(y1) * 12 + month_map[m1.lower()[:3]]
            end = int(y2) * 12 + month_map[m2.lower()[:3]]
            months = max(0, end - start + 1)
            total_months += min(months, 60)
        except (KeyError, ValueError):
            continue
    if total_months > 0:
        return round(total_months / 12.0, 1)
    return 0.0

def ats_score(text: str, skills: list) -> dict:
    score = 5.0
    reasons = []
    text_lower = text.lower()
    words = len(text.split())
    if 300 <= words <= 900:
        score += 1.2
        reasons.append("✅ Good length (300-900 words) – ATS-friendly")
    elif words < 200:
        score -= 1.5
        reasons.append("❌ Too short – add more detail so the ATS has content to rank")
    else:
        score -= 0.4
        reasons.append("⚠️ Somewhat long – tighten to 1 page (junior) or 2 pages max")
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text):
        score += 0.5
        reasons.append("✅ Email present")
    else:
        score -= 1.0
        reasons.append("❌ Missing email – ATS and recruiters need it")
    if re.search(r"(\+?\d[\d\s\-]{7,})", text) or "linkedin.com" in text_lower:
        score += 0.4
        reasons.append("✅ Phone or LinkedIn present")
    else:
        reasons.append("⚠️ Add phone number and LinkedIn URL")
    standard_headings = [
        "work experience", "professional experience", "experience",
        "education", "skills", "technical skills", "projects",
        "summary", "professional summary", "certifications", "languages"
    ]
    found_headings = sum(1 for h in standard_headings if h in text_lower)
    if found_headings >= 4:
        score += 1.5
        reasons.append("✅ Standard ATS-friendly section headings detected")
    elif found_headings >= 2:
        score += 0.6
        reasons.append("⚠️ Some standard headings found – prefer exact names: Work Experience, Education, Skills")
    else:
        score -= 1.0
        reasons.append("❌ Missing standard headings. Use: Work Experience, Education, Skills (avoid creative titles)")
    if len(skills) >= 10:
        score += 1.0
        reasons.append(f"✅ Strong skills section ({len(skills)} keywords) – good for ATS matching")
    elif len(skills) >= 5:
        score += 0.5
        reasons.append(f"⚠️ Moderate skills listed ({len(skills)}) – add more exact keywords from job ads")
    else:
        score -= 0.6
        reasons.append("❌ Few skills detected – expand with exact terms from target job descriptions")
    action_verbs = [
        "developed", "led", "managed", "implemented", "designed", "built",
        "created", "improved", "increased", "reduced", "achieved", "collaborated",
        "provided", "diagnosed", "installed", "configured", "supported", "assisted",
        "resolved", "maintained", "trained", "documented", "brewed", "served", "operated"
    ]
    verb_count = sum(1 for v in action_verbs if v in text_lower)
    if verb_count >= 6:
        score += 0.8
        reasons.append("✅ Strong action verbs used – helps both ATS and recruiters")
    else:
        reasons.append("⚠️ Use more action verbs (diagnosed, configured, resolved, improved, brewed...)")
    if text.count("|") > 8 or text.count("\t") > 15:
        score -= 1.0
        reasons.append("❌ Possible tables or complex formatting – ATS often fails to read tables/columns. Use single-column layout")
    else:
        score += 0.4
        reasons.append("✅ No heavy table/column formatting detected – good for ATS")
    if re.search(r"\d+%|\$\d+|\d+\s*(?:users|clients|projects|team|tickets|issues|customers)", text, re.I):
        score += 0.8
        reasons.append("✅ Quantifiable results present (numbers help ranking and human readers)")
    else:
        reasons.append("⚠️ Add measurable impact (e.g. “Reduced resolution time by 40%”, “Served 50+ customers daily”)")
    intern_count = len(re.findall(r"\bintern\b|\btrainee\b", text_lower))
    if intern_count >= 1:
        reasons.append("⚠️ Training/internship experience – strengthen measurable results and certifications")
        score -= 0.2
    if "graduate" in text_lower or "bachelor" in text_lower:
        reasons.append("ℹ️ Recent graduate / early-career profile – target Junior / Entry-level roles")
    if re.search(r"\b(senior|lead|principal|manager)\b", text_lower[:450]):
        if intern_count >= 1 or "graduate" in text_lower or "trainee" in text_lower:
            reasons.append("❌ Header uses advanced titles while experience is limited. Use Junior / Entry-level titles for credibility")
            score -= 0.9
    reasons.append("💡 Tip: Copy-paste your CV into Notepad. If it stays clean and readable, the ATS will most likely parse it correctly.")
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

def generate_cover_letter(cv_text, job_title, company, job_desc, skills, style="Professional"):
    name_match = re.search(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", cv_text)
    name = name_match.group(1) if name_match else "Applicant"
    relevant_skills = ", ".join(skills[:8]) if skills else "relevant skills"
    job_words = set(re.findall(r"\b\w{4,}\b", job_desc.lower()))
    sentences = re.split(r"[.!?]\s+", cv_text)
    scored = []
    for s in sentences:
        if len(s) < 30: continue
        overlap = len(set(re.findall(r"\b\w{4,}\b", s.lower())) & job_words)
        if overlap > 1: scored.append((overlap, s.strip()))
    scored.sort(reverse=True)
    highlights = " ".join([s[1] for s in scored[:3]]) if scored else "my professional experience and achievements."
    company_part = f" at {company}" if company else ""
    title_part = job_title or "the position"
    if style == "Professional":
        letter = f"""Dear Hiring Manager{(' at ' + company) if company else ''},

I am writing to express my strong interest in the {title_part} role{company_part}. With a background that includes {relevant_skills}, I am confident that my experience aligns well with the requirements of this role.

{highlights}

I would welcome the opportunity to discuss how my skills and experience can benefit your team. Thank you for considering my application.

Sincerely,
{name}
"""
    elif style == "Enthusiastic":
        letter = f"""Dear Hiring Team{(' at ' + company) if company else ''},

I am excited to apply for the {title_part} position{company_part}! My skills in {relevant_skills} make me a strong fit.

{highlights}

Thank you for your time and consideration — I would love the chance to chat further!

Best regards,
{name}
"""
    else:
        letter = f"""Dear Hiring Manager,

I am applying for the {title_part} role{company_part}. My background includes {relevant_skills}.

Key highlights:
{highlights}

I am confident I can add value quickly and would welcome the opportunity to discuss this further.

Sincerely,
{name}
"""
    return letter.strip()

def generate_linkedin_about(cv_text, skills, years):
    skill_str = ", ".join(skills[:10]) if skills else "customer service and professional skills"
    if years >= 5:
        level_phrase = f"Experienced professional with {years:.0f}+ years"
    elif years >= 1.5:
        level_phrase = f"Professional with {years:.1f} years of hands-on experience"
    elif years >= 0.3:
        level_phrase = f"Motivated professional with practical training and hands-on experience"
    else:
        level_phrase = "Motivated and dedicated professional"
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", cv_text) if 40 < len(s.strip()) < 180]
    highlights = " ".join(sentences[:2]) if sentences else "I focus on delivering excellent service and continuous learning."
    about = f"""{level_phrase} specializing in {skill_str}.

{highlights}

I am passionate about continuous learning and delivering high-quality results. Always open to Junior / Entry-level opportunities where I can contribute and grow.

Core strengths: {skill_str}.
"""
    return about.strip()

def suggest_job_titles(skills: list, years: float) -> list:
    skill_set = set(s.lower() for s in skills)
    suggestions = []
    def has(*keys):
        return any(k in skill_set for k in keys)
    if years >= 5: level = "Senior "
    elif years >= 2: level = "Mid-level "
    elif years >= 0.8: level = "Junior "
    else: level = "Entry-level / Junior "

    if has("barista", "espresso", "coffee", "beverage", "latte", "cappuccino", "cafe",
           "hospitality", "food service", "waiter", "waitress", "server", "chef", "cook", "kitchen"):
        suggestions.extend(["Barista", "Junior Barista", "Café Barista", "Coffee Shop Barista",
            "Beverage Specialist", "Café Assistant", "Food & Beverage Assistant",
            "Hospitality Assistant", "Waiter / Waitress", "Restaurant Server",
            "Kitchen Assistant", "Cashier (Hospitality)"])
    if has("windows", "linux", "networking", "tcp/ip", "dns", "dhcp", "end-user", "troubleshooting", "incident"):
        suggestions.extend([f"{level}IT Support Specialist", f"{level}Technical Support Engineer",
            f"{level}ICT Support Officer", "Help Desk Analyst", "Desktop Support Technician", "Service Desk Analyst"])
    if has("sql", "mysql", "database"):
        suggestions.extend([f"{level}Database Support Specialist", "Junior Database Administrator", "SQL Support Analyst"])
    if has("oracle"): suggestions.append("Oracle Support Engineer (Junior)")
    if has("php", "javascript", "html", "css", "mysql", "react", "node"):
        suggestions.extend([f"{level}Web Developer", "Junior PHP Developer", "Full Stack Developer (Junior)"])
    if has("python", "java", "c++", "software"):
        suggestions.extend([f"{level}Software Developer", f"{level}Software Engineer"])
    if has("aws", "azure", "docker", "kubernetes", "devops", "cloud"):
        suggestions.extend([f"{level}DevOps Engineer", "Cloud Support Associate", "Junior Cloud Engineer"])
    if has("networking", "cisco", "lan/wan"):
        suggestions.extend(["Network Support Technician", "Junior Network Administrator"])
    if has("vmware", "virtualbox", "linux"): suggestions.append("Junior Systems Administrator")
    if has("cybersecurity", "security"):
        suggestions.extend(["Cybersecurity Analyst (Junior)", "Security Operations Analyst"])
    if has("nursing", "nurse", "patient care", "clinical", "medical", "healthcare", "hospital", "clinic"):
        suggestions.extend(["Nursing Assistant", "Patient Care Assistant", "Clinical Support Worker",
            "Healthcare Assistant", "Medical Receptionist", "Hospital Ward Assistant"])
    if has("pharmacy", "pharmacist"): suggestions.extend(["Pharmacy Assistant", "Pharmacy Technician"])
    if has("lab", "phlebotomy", "radiology"):
        suggestions.extend(["Laboratory Assistant", "Phlebotomist", "Radiology Assistant"])
    if has("teaching", "teacher", "tutor", "curriculum", "classroom", "education", "training", "facilitation"):
        suggestions.extend(["Teacher", "Tutor", "Teaching Assistant", "Education Assistant",
            "Trainer / Facilitator", "Curriculum Support Officer", "Learning Support Assistant"])
    if has("accounting", "bookkeeping", "financial reporting", "audit", "taxation", "budgeting",
           "accounts payable", "accounts receivable", "payroll", "reconciliation"):
        suggestions.extend([f"{level}Accountant", "Accounts Assistant", "Bookkeeper",
            "Accounts Payable / Receivable Clerk", "Finance Assistant", "Audit Assistant"])
    if has("banking", "teller", "credit", "loan"):
        suggestions.extend(["Bank Teller", "Banking Officer", "Credit Analyst (Junior)", "Loan Officer Assistant"])
    if has("sales", "retail", "merchandising", "inventory", "stock", "crm", "lead generation"):
        suggestions.extend(["Sales Associate", "Retail Sales Associate", "Sales Representative",
            "Merchandiser", "Store Assistant", "Customer Service Associate (Retail)"])
    if has("marketing", "seo", "digital marketing", "social media", "content marketing", "branding"):
        suggestions.extend(["Marketing Assistant", "Digital Marketing Assistant", "Social Media Coordinator",
            "Content Marketing Assistant", "SEO Assistant"])
    if has("administration", "office management", "receptionist", "front desk", "secretarial",
           "data entry", "scheduling", "ms office", "microsoft office", "filing"):
        suggestions.extend(["Administrative Assistant", "Office Assistant", "Receptionist",
            "Front Desk Officer", "Data Entry Clerk", "Executive Assistant (Junior)", "Office Administrator"])
    if has("logistics", "supply chain", "warehouse", "procurement", "shipping", "freight", "inventory"):
        suggestions.extend(["Logistics Assistant", "Warehouse Assistant", "Store Keeper",
            "Procurement Assistant", "Supply Chain Coordinator (Junior)"])
    if has("driver", "driving", "delivery", "fleet", "transport"):
        suggestions.extend(["Delivery Driver", "Driver", "Fleet Assistant", "Transport Coordinator"])
    if has("graphic design", "design", "photoshop", "illustrator", "indesign", "canva", "photography"):
        suggestions.extend(["Graphic Designer (Junior)", "Design Assistant", "Creative Assistant",
            "Photographer Assistant", "Visual Content Creator"])
    if has("content creation", "social media management", "copywriting", "writing", "journalism"):
        suggestions.extend(["Content Creator", "Social Media Assistant", "Copywriter (Junior)",
            "Content Writer", "Communications Assistant"])
    if has("mechanical", "electrical", "civil", "engineering", "autocad", "solidworks", "technician"):
        suggestions.extend([f"{level}Mechanical Technician", f"{level}Electrical Technician",
            "Engineering Assistant", "CAD Technician", "Maintenance Technician"])
    if has("welding", "plumbing", "construction", "maintenance"):
        suggestions.extend(["Welder", "Plumber", "Construction Assistant", "Maintenance Assistant"])
    if has("security", "guard", "surveillance", "cctv", "safety", "hse", "fire safety"):
        suggestions.extend(["Security Guard", "Security Officer", "CCTV Operator",
            "Safety Officer (Junior)", "HSE Assistant"])
    if not suggestions and has("customer service", "communication", "teamwork"):
        suggestions.extend(["Customer Service Representative", "Front Desk / Receptionist",
            "Retail Sales Associate", "Call Centre Agent", f"{level}Customer Support"])
    if not suggestions:
        suggestions = [f"{level}Professional", "Customer Service Associate",
            "Administrative Assistant", "Entry-level Assistant"]
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
                if sn: snippet = sn.get_text(strip=True)
            if title: results.append({"title": title, "link": link, "snippet": snippet})
    except Exception:
        pass
    return results

def build_analysis_report(cv_text, skills, ats, years, suggestions) -> str:
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
{years:.1f} years

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

if "cv_text" not in st.session_state: st.session_state.cv_text = ""
if "skills" not in st.session_state: st.session_state.skills = []
if "ats" not in st.session_state: st.session_state.ats = None
if "years" not in st.session_state: st.session_state.years = 0

st.title("📄 CV RAG Job Assistant")
st.markdown("Upload your CV → get **ATS score**, **strengths & weaknesses**, **job match rating**, **tailored cover letters**, **LinkedIn About**, and **related job suggestions** across all major careers.")

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

if not st.session_state.cv_text:
    st.info("👈 Please upload your CV (PDF or Word) in the sidebar to get started.")
    st.markdown("""
        ### What this tool does
        | Feature | Description |
        |---------|-------------|
        | **ATS Rating** | Scores your CV 1-10 for Applicant Tracking Systems |
        | **Downfalls & Advice** | Points out weaknesses and how to improve |
        | **Job Match** | Rate your CV against any job description |
        | **Cover Letter** | Generate tailored letters in 3 styles |
        | **LinkedIn About** | Create a ready-to-use LinkedIn summary |
        | **Job Suggestions** | Roles that match your skills + experience (all careers) |
        | **Online Job Search** | Google / LinkedIn / Indeed / BrighterMonday links |
        | **Export Report** | Download full analysis as Markdown |
        """)
else:
    cv = st.session_state.cv_text
    skills = st.session_state.skills
    ats = st.session_state.ats
    years = st.session_state.years
    suggestions = suggest_job_titles(skills, years)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 CV Analysis & ATS", "🎯 Match vs Job", "✉️ Cover Letter",
        "🔗 LinkedIn About", "💼 Suggested Jobs", "🔍 Search Live Jobs"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("ATS Score", f"{ats['score']} / 10")
        with col2: st.metric("Detected Skills", len(skills))
        with col3: st.metric("Est. Experience", f"{years:.1f} years" if years else "Not clear")
        st.subheader("Score Breakdown & Advice")
        for r in ats["reasons"]: st.write(r)
        st.subheader("Detected Skills")
        if skills: st.write(", ".join(skills))
        else: st.warning("Few skills detected. Add a clear Skills section with keywords.")
        st.subheader("⚠️ Potential Downfalls")
        downs = [r for r in ats["reasons"] if r.startswith("❌") or r.startswith("⚠️")]
        if downs:
            for d in downs: st.write(d)
        else: st.success("No major red flags detected!")
        st.subheader("💡 Quick Improvement Tips")
        st.markdown("""
            - Use standard section headers: **Summary, Experience, Education, Skills**
            - Start bullets with strong action verbs + numbers
            - Mirror keywords from the job description you are applying to
            - Keep formatting simple (no tables, text boxes, or heavy graphics)
            - Aim for 1 page (junior) or 2 pages (senior)
            - Include LinkedIn / portfolio if relevant
            """)
        st.subheader("CV Preview (first 1500 chars)")
        st.text_area("Extracted text", cv[:1500] + ("..." if len(cv) > 1500 else ""), height=200)
        report = build_analysis_report(cv, skills, ats, years, suggestions)
        st.download_button("📥 Download Full Analysis Report (.md)", report, file_name="cv_analysis_report.md", mime="text/markdown")

    with tab2:
        st.subheader("Rate your CV against a specific job")
        job_title = st.text_input("Job Title", placeholder="e.g. Barista / Junior IT Support", key="match_title")
        company = st.text_input("Company (optional)", placeholder="e.g. Acme Corp", key="match_company")
        job_desc = st.text_area("Paste the full Job Description here", height=220, placeholder="Paste the requirements...", key="match_desc")
        if st.button("🔍 Analyze Match", type="primary") and job_desc.strip():
            with st.spinner("Computing similarity (RAG)..."):
                sim = similarity_score(cv, job_desc)
                match_score = round(sim * 10, 1)
                missing = missing_keywords(cv, job_desc)
            st.metric("Match Score (CV ↔ Job)", f"{match_score} / 10")
            st.progress(min(sim, 1.0))
            if match_score >= 7.5: st.success("Strong match!")
            elif match_score >= 5: st.warning("Moderate match. Tailor your CV and cover letter.")
            else: st.error("Low match. Consider adding missing keywords or a better-fitting role.")
            st.subheader("Missing / Under-emphasized Keywords")
            if missing: st.write(", ".join(missing))
            else: st.write("No major missing keywords detected.")
            st.subheader("Most Relevant Parts of Your CV (RAG)")
            job_words = set(re.findall(r"\b\w{4,}\b", job_desc.lower()))
            sentences = [s.strip() for s in re.split(r"[.!?]\s+", cv) if len(s.strip()) > 40]
            scored = []
            for s in sentences:
                overlap = len(set(re.findall(r"\b\w{4,}\b", s.lower())) & job_words)
                scored.append((overlap, s))
            scored.sort(reverse=True)
            for _, s in scored[:5]: st.write(f"• {s}")

    with tab3:
        st.subheader("Generate Application / Cover Letter")
        cl_job = st.text_input("Job Title for letter", key="cl_job")
        cl_company = st.text_input("Company name", key="cl_comp")
        cl_desc = st.text_area("Job description (for better tailoring)", height=150, key="cl_desc")
        style = st.selectbox("Letter style", ["Professional", "Enthusiastic", "Concise"])
        if st.button("✍️ Generate Letter", type="primary"):
            letter = generate_cover_letter(cv, cl_job, cl_company, cl_desc or "the role", skills, style=style)
            st.text_area("Your tailored cover letter", letter, height=420)
            st.download_button("Download as .txt", letter, file_name="cover_letter.txt")

    with tab4:
        st.subheader("Generate LinkedIn About / Summary")
        st.caption("A ready-to-paste professional summary for your LinkedIn profile.")
        if st.button("🔗 Generate LinkedIn About", type="primary"):
            about = generate_linkedin_about(cv, skills, years)
            st.text_area("LinkedIn About section", about, height=280)
            st.download_button("Download as .txt", about, file_name="linkedin_about.txt")

    with tab5:
        st.subheader("Roles you can apply for (based on your CV)")
        for i, title in enumerate(suggestions, 1):
            st.write(f"{i}. **{title}**")
        st.markdown("---")
        st.subheader("How to use these suggestions")
        st.markdown("""
            1. Search these titles on LinkedIn, Indeed, Glassdoor, BrighterMonday, etc.
            2. For each interesting posting, paste the job description into the **Match vs Job** tab.
            3. Improve your CV with the missing keywords, then generate a fresh cover letter.
            """)

    with tab6:
        st.subheader("Search available jobs (Google, LinkedIn, Indeed...)")
        st.caption("Automated scraping is often blocked. Use the ready-made links below – they open real job search results based on your CV skills.")
        default_query = suggestions[0] if suggestions else "Junior professional"
        q = st.text_input("Search query", value=default_query, key="search_q")
        loc = st.text_input("Location preference", value="Kenya OR Nairobi OR remote", key="search_loc")
        q_enc = q.replace(" ", "+")
        q_li = q.replace(" ", "%20")
        st.markdown(f"""
**Direct job search links (based on your CV):**
- [Google Jobs – “{q}”](https://www.google.com/search?q={q_enc}+jobs+{loc.replace(" ", "+")}&ibp=htl;jobs)
- [LinkedIn – “{q}”](https://www.linkedin.com/jobs/search/?keywords={q_li}&location={loc.replace(" ", "%20")})
- [Indeed – “{q}”](https://www.indeed.com/jobs?q={q_enc}&l={loc.replace(" ", "+")})
- [BrighterMonday Kenya – “{q}”](https://www.brightermonday.co.ke/jobs?q={q_enc})
- [Glassdoor – “{q}”](https://www.glassdoor.com/Job/jobs.htm?sc.keyword={q_enc})
""")
        if st.button("🔎 Try automated search (best-effort)"):
            with st.spinner("Searching (may be blocked by search engines)..."):
                jobs = search_jobs_online(q, loc)
            if jobs:
                st.success(f"Found {len(jobs)} results")
                for j in jobs:
                    st.markdown(f"**[{j['title']}]({j['link']})**")
                    if j["snippet"]: st.caption(j["snippet"][:220])
                    st.write("---")
            else:
                st.warning("Automated search returned nothing (common). Please use the Google / LinkedIn / Indeed / BrighterMonday links above.")

st.markdown("---")
st.caption("Built with Streamlit + RAG. Heuristic ATS scoring – always review the output manually before applying.")
