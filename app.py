from flask import Flask, render_template, request, jsonify
import anthropic
import json
import re
import io
import difflib
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import docx

app = Flask(__name__)

# ── ADJACENCY MAP (semantic matching) ────────────────────────────────────────
ADJACENCY_MAP = {
    'gradient boosting': ['xgboost','lightgbm','catboost','gbm'],
    'deep learning': ['neural network','cnn','rnn','lstm','transformer','pytorch','tensorflow','keras'],
    'llm agents': ['langchain','llamaindex','crewai','autogen','agentic ai','rag'],
    'rest api': ['flask','fastapi','django','express','api development'],
    'data manipulation': ['pandas','numpy','polars','dplyr'],
    'cloud computing': ['aws','azure','gcp','s3','ec2','lambda','sagemaker'],
    'data visualization': ['tableau','power bi','matplotlib','seaborn','plotly','looker'],
    'version control': ['git','github','gitlab','bitbucket'],
    'containerization': ['docker','kubernetes','k8s'],
    'nlp': ['natural language processing','text mining','nltk','spacy','bert','transformers'],
    'statistical modeling': ['regression','classification','hypothesis testing','a/b testing','statistics'],
    'etl pipeline': ['data pipeline','airflow','data engineering','dbt','spark'],
    'database': ['sql','mysql','postgresql','mongodb','nosql','sqlite'],
}

SYSTEM_PROMPT = """You are an enterprise-grade ATS (Applicant Tracking System) analyzer. Analyze the resume against the job description using this strict 7-stage pipeline:

CANDIDATE PROFILE — Satya Praneeth Mallipam:
- MS Data Science, Kent State University, graduating May 2026, GPA 3.71
- F-1 OPT (needs H-1B sponsorship ~2029)
- Academic projects only — no paid work experience
- Springer-published ML paper (premature birth, 87% F1, 11K records)
- Projects: Business Feasibility Prediction (XGBoost/Flask/React/30K ZIPs), Hospital ER Wait Time (SQL/Tableau/50K records), ML Premature Birth (XGBoost/H2O AutoML)
- Skills: Python, SQL, R, Tableau, Power BI, Excel, XGBoost, scikit-learn, Pandas, NumPy, ETL, AWS, ML, Flask, React
- Certs: AWS Academy (Data Engineering, ML Foundations, Cloud Architecting), Cisco

STAGE 1 — HARD ELIGIBILITY GATE:
Check: graduation window, sponsorship, clearance/citizenship, degree level, location.
One FAIL = DO NOT APPLY regardless of other scores.

STAGE 2 — PARSE QUALITY:
Pre-computed rule-based score will be provided. Factor it in.

STAGE 3 — REQUIRED KEYWORD SCORING (weight 3x for technical, 2x domain, 1x soft skills):
Extract top 12-15 required keywords from JD. For each found keyword, check:
- Is it ONLY in a comma-separated skills list? → score 0.5x (list-only penalty)
- Is it also in a project bullet point with context? → score 1.0x (full credit)
Report contextual_validation for each keyword.

STAGE 4 — PREFERRED KEYWORD SCORING (adjacency mapping):
Score nice-to-have qualifications. Award 0.8x partial credit for adjacent technologies
(e.g. XGBoost counts partially toward "gradient boosting", Flask counts toward "REST API").

STAGE 5 — SEMANTIC PERSONA MATCH:
Assess holistic fit: career level, domain depth, persona alignment.

STAGE 6 — IMPACT QUANTIFICATION:
Count bullet points WITH measurable metrics (numbers, %, scale) vs WITHOUT.
Bullets without metrics score 0.7x. Report ratio.

STAGE 7 — COMPOSITE SCORE:
parse*0.15 + required*0.35 + preferred*0.15 + semantic*0.15 + recruiter*0.10 + hiringMgr*0.10

Respond ONLY with valid JSON (no markdown, no backticks, no preamble):
{
  "composite": <0-100>,
  "verdict": "<SUBMIT IMMEDIATELY | SUBMIT WITH EDITS | REVISE SIGNIFICANTLY | DO NOT APPLY>",
  "verdictSub": "<one sentence>",
  "scores": {
    "parse": <0-100>,
    "required": <0-100>,
    "preferred": <0-100>,
    "semantic": <0-100>,
    "recruiter": <0-100>,
    "hiringMgr": <0-100>
  },
  "gates": [{"name":"<gate>","pass":<bool>,"note":"<if fail>"}],
  "keywords": [{"term":"<keyword>","found":<bool>,"context":"<list_only|in_project|missing>","weight":<0.5|0.8|1.0>}],
  "contextual_validation": [{"skill":"<skill>","list_only":<bool>,"in_project":<bool>,"penalty":"<none|0.5x>"}],
  "impact": {"quantified_bullets":<int>,"total_bullets":<int>,"ratio":<0.0-1.0>,"weak_bullets":["<bullet needing metric>"]},
  "adjacency_matches": [{"jd_term":"<term>","resume_term":"<term>","credit":0.8}],
  "edits": [{"title":"<title>","description":"<specific LaTeX instruction>"}],
  "summary": "<2-3 sentence assessment>"
}"""

REWRITE_SYSTEM_PROMPT = """You are an expert LaTeX resume editor for Satya Praneeth Mallipam.

Given: Main.tex content, job description, ATS analysis results.

Generate EXACT find-and-replace patches. Rules:
- FIND must be verbatim substring from Main.tex (SHORT — 1-2 lines max)
- REPLACE must be valid LaTeX
- Do NOT fabricate skills/experience
- DO add metrics to weak bullet points (use realistic numbers from project context)
- DO reword bullets to include missing keywords naturally
- DO strengthen list-only skills by adding project context
- Generate 5-10 patches

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "patches": [
    {
      "id": <int>,
      "section": "<Summary|Skills|Project 1|Project 2|Project 3|Education|Publication>",
      "reason": "<why this helps + which JD keyword it targets>",
      "find": "<exact verbatim 1-2 line substring from Main.tex>",
      "replace": "<new LaTeX>"
    }
  ],
  "summary": "<2-3 sentence rewrite strategy>",
  "projected_score": <0-100>
}"""


# ── PDF EXTRACTION ────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes):
    output = io.StringIO()
    laparams = LAParams(line_margin=0.5, word_margin=0.1, char_margin=2.0, boxes_flow=0.5, detect_vertical=False)
    extract_text_to_fp(io.BytesIO(file_bytes), output, laparams=laparams, output_type='text', codec='utf-8')
    raw = output.getvalue()
    raw = re.sub(r'\x00', '', raw)
    raw = re.sub(r'\u00ad', '', raw)
    raw = re.sub(r'[\x01-\x08\x0b\x0e-\x1f]', '', raw)
    raw = re.sub(r'[ \t]{3,}', '  ', raw)
    raw = re.sub(r'\n{4,}', '\n\n', raw)
    return raw.strip()


def extract_text_from_file(file):
    filename = file.filename.lower()
    file_bytes = file.read()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file_bytes))
        return " ".join(para.text for para in doc.paragraphs)
    return ""


# ── RULE-BASED PARSE QUALITY ──────────────────────────────────────────────────

SECTION_PATTERNS = {
    'education':   re.compile(r'^(education|academic|qualification)', re.I | re.M),
    'skills':      re.compile(r'^(skills?|technical|competencies|technologies)', re.I | re.M),
    'projects':    re.compile(r'^(projects?|portfolio)', re.I | re.M),
}

def rule_based_parse_quality(text):
    issues = []
    score = 100
    if '\u00ad' in text:
        issues.append("Soft hyphens (U+00AD) detected — breaks keyword parsing")
        score -= 25
    ctrl = re.findall(r'[\x01-\x08\x0b\x0e-\x1f]', text)
    if ctrl:
        issues.append(f"{len(ctrl)} control characters found")
        score -= 15
    for s, pat in SECTION_PATTERNS.items():
        if not pat.search(text):
            issues.append(f"Section '{s}' not detected by parser")
            score -= 10
    key_phrases = ['machine learning','data science','data analysis','data engineering']
    for phrase in key_phrases:
        if re.search(phrase[:len(phrase)//2] + r'\s*\n\s*' + phrase[len(phrase)//2:], text, re.I):
            issues.append(f"'{phrase}' broken across lines")
            score -= 5
    if len(text) < 200:
        issues.append("Very little text — PDF may be image-based")
        score -= 40
    return max(0, score), issues


# ── PATCH VALIDATION ──────────────────────────────────────────────────────────

def validate_patches(patches, tex_content):
    tex_lines = tex_content.splitlines()
    validated = []
    for patch in patches:
        find_str = patch.get('find', '')
        p = dict(patch)
        if not find_str:
            p['valid'] = False; p['validation_msg'] = 'Empty FIND string'
            validated.append(p); continue
        if find_str in tex_content:
            p['valid'] = True; p['validation_msg'] = 'Exact match ✓'
            validated.append(p); continue
        matches = difflib.get_close_matches(find_str.strip(), [l.strip() for l in tex_lines if l.strip()], n=1, cutoff=0.6)
        if matches:
            p['valid'] = False
            p['validation_msg'] = f'Not found exactly — closest: "{matches[0][:80]}"'
            p['suggested_find'] = matches[0]
        else:
            p['valid'] = False
            p['validation_msg'] = 'Not found in file — verify manually'
        validated.append(p)
    return validated


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        jd_text = request.form.get("jd", "").strip()
        resume_text = ""
        parse_issues = []
        rule_parse_score = None

        if "resumeFile" in request.files and request.files["resumeFile"].filename:
            try:
                resume_text = extract_text_from_file(request.files["resumeFile"])
                if resume_text:
                    rule_parse_score, parse_issues = rule_based_parse_quality(resume_text)
            except Exception as e:
                return jsonify({"error": f"File parse error: {str(e)}"}), 400
        elif request.is_json:
            data = request.json
            resume_text = data.get("resume", "").strip()
            jd_text = data.get("jd", "").strip()

        if not resume_text:
            return jsonify({"error": "Could not extract text from file."}), 400
        if not jd_text:
            return jsonify({"error": "Job description is required."}), 400

        parse_context = ""
        if parse_issues:
            parse_context = f"\n\nPRE-COMPUTED PARSE ANALYSIS:\n- Issues: {'; '.join(parse_issues)}\n- Rule-based parse score: {rule_parse_score}/100\nUse this for Stage 2."

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB DESCRIPTION:\n{jd_text}{parse_context}\n\nRun all 7 stages. Return only JSON."}]
        )

        raw = "".join(b.text for b in message.content if hasattr(b, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)

        if rule_parse_score is not None:
            result['scores']['parse'] = round(0.6 * rule_parse_score + 0.4 * result['scores'].get('parse', rule_parse_score))
            s = result['scores']
            result['composite'] = round(s['parse']*0.15 + s['required']*0.35 + s['preferred']*0.15 + s['semantic']*0.15 + s['recruiter']*0.10 + s['hiringMgr']*0.10)

        if parse_issues:
            result['parse_issues'] = parse_issues

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rewrite", methods=["POST"])
def rewrite():
    try:
        jd_text = request.form.get("jd", "").strip()
        analysis_json = request.form.get("analysis", "{}").strip()

        if "texFile" not in request.files or not request.files["texFile"].filename:
            return jsonify({"error": "Please upload Main.tex"}), 400

        tex_file = request.files["texFile"]
        if not tex_file.filename.lower().endswith(".tex"):
            return jsonify({"error": "Only .tex files accepted"}), 400

        tex_content = tex_file.read().decode("utf-8")
        if not jd_text:
            return jsonify({"error": "Job description required. Run analysis first."}), 400

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=3000,
            system=REWRITE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"MAIN.TEX:\n```latex\n{tex_content}\n```\n\n---\n\nJOB DESCRIPTION:\n{jd_text}\n\n---\n\nATS RESULTS:\n{analysis_json}\n\nGenerate patches. FIND strings must be SHORT (1-2 lines) and verbatim. Return only JSON."}]
        )

        raw = "".join(b.text for b in message.content if hasattr(b, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)

        if 'patches' in result:
            result['patches'] = validate_patches(result['patches'], tex_content)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    print("\n✅ ATS Analyzer running at http://localhost:5000\n")
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)), host='0.0.0.0', use_reloader=False)
