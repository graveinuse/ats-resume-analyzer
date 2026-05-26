from flask import Flask, render_template, request, jsonify
import anthropic
import json
import re
import io
import PyPDF2
import docx

app = Flask(__name__)

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
1. The current LaTeX content of Main.tex (the candidate's resume)
2. A job description
3. ATS analysis results showing missing keywords and suggested improvements

Your job is to generate EXACT find-and-replace patches to improve the resume for this specific job.

RULES:
- Only modify Main.tex content (not Settings.cls or Resume.tex)
- Every FIND string must exist VERBATIM in the provided Main.tex — copy it exactly, character for character
- Every REPLACE string must be valid LaTeX that compiles without errors
- Do NOT fabricate experience or skills the candidate doesn't have
- DO reword existing bullet points to include missing keywords naturally
- DO add missing skills to the Technical Skills section if candidate genuinely has them
- DO strengthen project descriptions with JD-relevant terminology
- Keep all changes honest and defensible in an interview
- Generate 5-10 targeted patches

Respond ONLY with valid JSON (no markdown, no backticks, no preamble):
{
  "patches": [
    {
      "id": <number>,
      "section": "<which section: Summary | Skills | Project 1 | Project 2 | Project 3 | Education | Publication>",
      "reason": "<one sentence: why this change helps, which JD keyword it targets>",
      "find": "<exact verbatim string from Main.tex — must match character for character>",
      "replace": "<new LaTeX string to replace it with>"
    }
  ],
  "summary": "<2-3 sentence explanation of overall strategy for this resume rewrite>",
  "projected_score": <estimated new composite score after patches, number 0-100>
}"""


def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file.read()))
        return " ".join(para.text for para in doc.paragraphs)
    else:
        return ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        jd_text = request.form.get("jd", "").strip()
        resume_text = ""

        if "resumeFile" in request.files and request.files["resumeFile"].filename:
            try:
                resume_text = extract_text_from_file(request.files["resumeFile"])
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

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"RESUME TEXT:\n{resume_text}\n\n---\n\nJOB DESCRIPTION:\n{jd_text}\n\nRun the full 7-stage ATS analysis. Return only the JSON object."
            }]
        )

        raw = "".join(b.text for b in message.content if hasattr(b, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
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
                    "Generate the exact find-and-replace patches to maximally improve this resume for the job. "
                    "Every FIND string must exist verbatim in the LaTeX above. Return only the JSON object."
                )
            }]
        )

        raw = "".join(b.text for b in message.content if hasattr(b, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n✅ ATS Analyzer running at http://localhost:5000\n")
    app.run(debug=True, port=5000)