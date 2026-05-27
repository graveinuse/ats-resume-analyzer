from flask import Flask, render_template, request, jsonify
import anthropic
import json
import re
import io
import difflib

# pdfminer for better PDF extraction
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_text

import docx

app = Flask(__name__)

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ATS resume analyzer for a specific candidate: Satya Praneeth Mallipam.

Candidate profile:
- MS Data Science, Kent State University, graduating May 2026, GPA 3.71
- F-1 OPT work authorization (needs H-1B sponsorship in future ~2029)
- No formal paid work experience — all academic projects
- Springer-published ML paper (premature birth prediction, 87% F1, 11K records)
- Projects: Business Feasibility Prediction (XGBoost/Flask/React/30K ZIPs), Hospital ER Wait Time (SQL/Tableau/50K records), ML Premature Birth (XGBoost/H2O AutoML, Springer)
- Skills: Python, SQL, R, Tableau, Power BI, Excel, XGBoost, scikit-learn, Pandas, NumPy, ETL, AWS, statistical modeling, ML, Flask, React
- Certifications: AWS Academy (Data Engineering, ML Foundations, Cloud Architecting), Cisco Python, Cisco Cybersecurity

Respond ONLY with valid JSON (no markdown, no backticks, no preamble):
{
  "composite": <number 0-100>,
  "verdict": "<SUBMIT IMMEDIATELY | SUBMIT WITH EDITS | REVISE SIGNIFICANTLY | DO NOT APPLY>",
  "verdictSub": "<one sentence explanation>",
  "scores": {
    "parse": <number>,
    "required": <number>,
    "preferred": <number>,
    "semantic": <number>,
    "recruiter": <number>,
    "hiringMgr": <number>
  },
  "gates": [{"name":"<gate>","pass":<bool>,"note":"<brief note if fail, else empty string>"}],
  "keywords": [{"term":"<keyword>","found":<bool>}],
  "edits": [{"title":"<edit title>","description":"<specific LaTeX Main.tex find-and-replace instruction>"}],
  "summary": "<2-3 sentence overall assessment>"
}

Gates to check: graduation window, sponsorship requirement, clearance/citizenship, degree level match, location.
Keywords: extract top 12-15 required keywords from JD, check each against candidate profile.
Edits: provide 3-5 specific actionable LaTeX edits for Main.tex to boost keyword match.
Composite = parse*0.15 + required*0.35 + preferred*0.15 + semantic*0.15 + recruiter*0.10 + hiringMgr*0.10"""

REWRITE_SYSTEM_PROMPT = """You are an expert LaTeX resume editor. You will be given:
1. The current LaTeX content of Main.tex
2. A job description
3. ATS analysis results showing missing keywords and suggested improvements

Generate EXACT find-and-replace patches to improve the resume for this specific job.

RULES:
- Only modify Main.tex content (not Settings.cls)
- Every FIND string must be a substring that EXISTS verbatim in the provided Main.tex
- Every REPLACE string must be valid LaTeX
- Do NOT fabricate experience or skills the candidate doesn't have
- DO reword existing bullet points to include missing keywords naturally
- DO add missing skills to Technical Skills section if candidate genuinely has them
- Keep all changes honest and defensible in an interview
- Generate 5-10 targeted patches
- FIND strings should be SHORT and specific (1-2 lines max) to ensure exact matching

Respond ONLY with valid JSON (no markdown, no backticks, no preamble):
{
  "patches": [
    {
      "id": <number>,
      "section": "<Summary | Skills | Project 1 | Project 2 | Project 3 | Education | Publication>",
      "reason": "<one sentence: why this change helps, which JD keyword it targets>",
      "find": "<short exact verbatim substring from Main.tex — 1-2 lines max>",
      "replace": "<new LaTeX string>"
    }
  ],
  "summary": "<2-3 sentence explanation of overall rewrite strategy>",
  "projected_score": <estimated new composite score after patches, 0-100>
}"""


# ── PDF EXTRACTION (pdfminer — much better than PyPDF2) ───────────────────────

def extract_text_from_pdf(file_bytes):
    """
    Uses pdfminer.six for layout-aware extraction.
    Handles multi-column, preserves reading order — much closer to real ATS parsing.
    """
    output = io.StringIO()
    laparams = LAParams(
        line_margin=0.5,
        word_margin=0.1,
        char_margin=2.0,
        boxes_flow=0.5,   # 0.5 = balanced horizontal+vertical flow (ATS-like)
        detect_vertical=False
    )
    extract_text_to_fp(
        io.BytesIO(file_bytes),
        output,
        laparams=laparams,
        output_type='text',
        codec='utf-8'
    )
    raw = output.getvalue()

    # Clean up common ATS-breaking artifacts
    raw = re.sub(r'\x00', '', raw)           # null bytes
    raw = re.sub(r'\u00ad', '', raw)         # soft hyphens (U+00AD) — breaks keywords
    raw = re.sub(r'[\x01-\x08\x0b\x0e-\x1f]', '', raw)  # control chars
    raw = re.sub(r'[ \t]{3,}', '  ', raw)   # excessive whitespace
    raw = re.sub(r'\n{4,}', '\n\n', raw)     # excessive blank lines

    return raw.strip()


def extract_text_from_file(file):
    filename = file.filename.lower()
    file_bytes = file.read()

    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file_bytes))
        return " ".join(para.text for para in doc.paragraphs)
    else:
        return ""


# ── ATS-STYLE SECTION DETECTION ──────────────────────────────────────────────

SECTION_PATTERNS = {
    'contact':     re.compile(r'(phone|email|linkedin|github|address)', re.I),
    'summary':     re.compile(r'^(summary|objective|profile|about)', re.I | re.M),
    'education':   re.compile(r'^(education|academic|qualification)', re.I | re.M),
    'experience':  re.compile(r'^(experience|employment|work history|positions?)', re.I | re.M),
    'skills':      re.compile(r'^(skills?|technical|competencies|technologies)', re.I | re.M),
    'projects':    re.compile(r'^(projects?|portfolio|work samples?)', re.I | re.M),
    'publications':re.compile(r'^(publications?|research|papers?)', re.I | re.M),
    'certifications': re.compile(r'^(certifications?|licenses?|credentials)', re.I | re.M),
}

def detect_sections(text):
    """Detect which resume sections are present — like a real ATS section parser."""
    found = {}
    for section, pattern in SECTION_PATTERNS.items():
        found[section] = bool(pattern.search(text))
    return found


def rule_based_parse_quality(text):
    """
    Rule-based parse quality score — checks what real ATS systems care about.
    Returns score 0-100 and list of issues found.
    """
    issues = []
    score = 100

    # Soft hyphens (major ATS killer)
    if '\u00ad' in text:
        issues.append("Soft hyphens (U+00AD) detected — breaks keyword parsing")
        score -= 25

    # Control characters
    ctrl_chars = re.findall(r'[\x01-\x08\x0b\x0e-\x1f]', text)
    if ctrl_chars:
        issues.append(f"{len(ctrl_chars)} control characters found")
        score -= 15

    # Check for key sections
    sections = detect_sections(text)
    missing_sections = [s for s, found in sections.items() if not found and s in ['education','skills']]
    if missing_sections:
        issues.append(f"Missing detectable sections: {', '.join(missing_sections)}")
        score -= 10 * len(missing_sections)

    # Multi-word keyword integrity (ATS breaks these if hyphenated oddly)
    key_phrases = ['machine learning', 'data science', 'deep learning',
                   'natural language', 'neural network', 'computer vision',
                   'data analysis', 'data engineering', 'business intelligence']
    for phrase in key_phrases:
        # Check if phrase appears broken across lines
        broken = re.search(phrase[:len(phrase)//2] + r'\s*\n\s*' + phrase[len(phrase)//2:], text, re.I)
        if broken:
            issues.append(f"'{phrase}' appears broken across lines")
            score -= 5

    # Very short text = extraction likely failed
    if len(text) < 200:
        issues.append("Very little text extracted — PDF may be image-based or corrupted")
        score -= 40

    return max(0, score), issues


# ── PATCH VALIDATION ─────────────────────────────────────────────────────────

def validate_patches(patches, tex_content):
    """
    Validates each patch FIND string against actual tex content.
    Uses fuzzy matching to find closest line if exact match fails.
    Returns patches with validation status added.
    """
    validated = []
    tex_lines = tex_content.splitlines()

    for patch in patches:
        find_str = patch.get('find', '')
        p = dict(patch)

        if not find_str:
            p['valid'] = False
            p['validation_msg'] = 'Empty FIND string'
            validated.append(p)
            continue

        # Exact match check
        if find_str in tex_content:
            p['valid'] = True
            p['validation_msg'] = 'Exact match ✓'
            validated.append(p)
            continue

        # Fuzzy match — find closest line
        find_stripped = find_str.strip()
        matches = difflib.get_close_matches(
            find_stripped,
            [l.strip() for l in tex_lines if l.strip()],
            n=1,
            cutoff=0.6
        )

        if matches:
            p['valid'] = False
            p['validation_msg'] = f'Not found exactly — closest match: "{matches[0][:80]}"'
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
                f = request.files["resumeFile"]
                resume_text = extract_text_from_file(f)

                # Run rule-based parse quality check (ATS simulation)
                if resume_text:
                    rule_parse_score, parse_issues = rule_based_parse_quality(resume_text)
            except Exception as e:
                return jsonify({"error": f"File parse error: {str(e)}"}), 400

        elif request.is_json:
            data = request.json
            resume_text = data.get("resume", "").strip()
            jd_text = data.get("jd", "").strip()

        if not resume_text:
            return jsonify({"error": "Could not extract text from file. Try a different PDF or DOCX."}), 400
        if not jd_text:
            return jsonify({"error": "Job description is required."}), 400

        # Pass parse issues to Claude so it factors them into score
        parse_context = ""
        if parse_issues:
            parse_context = f"\n\nRULE-BASED PARSE ANALYSIS (pre-computed):\n- Issues found: {'; '.join(parse_issues)}\n- Rule-based parse score: {rule_parse_score}/100\nFactor this into your parse quality score."

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"RESUME TEXT:\n{resume_text}\n\n"
                    f"---\n\nJOB DESCRIPTION:\n{jd_text}"
                    f"{parse_context}\n\n"
                    "Run the full 7-stage ATS analysis. Return only the JSON object."
                )
            }]
        )

        raw = "".join(b.text for b in message.content if hasattr(b, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)

        # Override parse score with rule-based result if available
        if rule_parse_score is not None:
            # Blend: 60% rule-based, 40% Claude's assessment
            result['scores']['parse'] = round(
                0.6 * rule_parse_score + 0.4 * result['scores'].get('parse', rule_parse_score)
            )
            # Recalculate composite
            s = result['scores']
            result['composite'] = round(
                s['parse']*0.15 + s['required']*0.35 + s['preferred']*0.15 +
                s['semantic']*0.15 + s['recruiter']*0.10 + s['hiringMgr']*0.10
            )

        # Add parse issues to result for display
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
            return jsonify({"error": "Please upload your Main.tex file."}), 400

        tex_file = request.files["texFile"]
        if not tex_file.filename.lower().endswith(".tex"):
            return jsonify({"error": "Only .tex files accepted."}), 400

        tex_content = tex_file.read().decode("utf-8")

        if not jd_text:
            return jsonify({"error": "Job description is required. Run the analysis first."}), 400

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=3000,
            system=REWRITE_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"MAIN.TEX CONTENT:\n```latex\n{tex_content}\n```\n\n"
                    f"---\n\nJOB DESCRIPTION:\n{jd_text}\n\n"
                    f"---\n\nATS ANALYSIS RESULTS:\n{analysis_json}\n\n"
                    "Generate exact find-and-replace patches. "
                    "Keep FIND strings SHORT (1-2 lines) and verbatim from the LaTeX above. "
                    "Return only the JSON object."
                )
            }]
        )

        raw = "".join(b.text for b in message.content if hasattr(b, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)

        # ── PATCH VALIDATION (eliminates hallucinated patches) ──
        if 'patches' in result:
            result['patches'] = validate_patches(result['patches'], tex_content)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n✅ ATS Analyzer running at http://localhost:5000\n")
    import os
    app.run(
        debug=True,
        port=int(os.environ.get('PORT', 5000)),
        host='0.0.0.0',
        use_reloader=False
    )