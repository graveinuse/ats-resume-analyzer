# ATS Resume Analyzer

7-stage ATS pipeline powered by Claude API. Runs locally in your browser.

## Setup (one time)

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Anthropic API key:
   ```
   # Mac/Linux:
   export ANTHROPIC_API_KEY=sk-ant-...

   # Windows (Command Prompt):
   set ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Open browser → http://localhost:5000

## Usage

1. Paste resume text (copy from your PDF)
2. Paste job description
3. Click "Run 7-Stage ATS Analysis"

## Output

- Composite ATS score (0-100)
- Verdict: SUBMIT IMMEDIATELY / SUBMIT WITH EDITS / REVISE / DO NOT APPLY
- Eligibility gate check (sponsorship, clearance, graduation window, etc.)
- Keyword match table (found vs missing)
- Score breakdown across 6 dimensions
- Specific LaTeX edits for Main.tex
- Overall assessment

## Get API Key

https://console.anthropic.com → API Keys → Create Key
