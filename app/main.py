from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from app.hybrid_matcher import hybrid_match_score

import os
import shutil
from typing import List
import re
import zipfile
import io

from app.resume_parser import extract_text_from_pdf
from app.preprocess import clean_resume_text
from app.gpt_matcher import get_resume_match_score
from app.cosine_matcher import cosine_match_score
from app.key_matcher import keyword_match_score

import fitz
import docx2txt
LAST_RESULTS = []

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="app/templates")


def extract_years(text):
    match = re.search(r'(\d+)[ ]*[-to]{0,3}[ ]*(\d+)?[ ]*years?', text.lower())
    if match:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        return (start + end) // 2
    return None


def enhanced_hybrid_score(gpt, cosine, keyword, resume_exp, jd_exp):
    score = (gpt * 0.4) + (cosine * 0.3) + (keyword * 0.3)
    exp_reason = ""

    if jd_exp and resume_exp is not None:
        if resume_exp >= jd_exp:
            score += 5
            exp_reason = f" Resume meets the required experience ({resume_exp} vs {jd_exp})."
        else:
            score -= 5
            exp_reason = f" Resume lacks required experience ({resume_exp} vs {jd_exp})."

    return max(0, min(100, round(score, 2))), exp_reason


@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("extract.html", {"request": request})


@app.get("/download-select", response_class=HTMLResponse)
async def download_select(request: Request):
    return templates.TemplateResponse("download_form.html", {"request": request})


@app.post("/download_zip")
async def download_cvs(request: Request, top_n: int = Form(...), zip_name: str = Form("")):
    zip_filename = zip_name.strip() or "top_resumes.zip"
    if not zip_filename.endswith(".zip"):
        zip_filename += ".zip"

    selected = LAST_RESULTS[:top_n]

    # Create in-memory zip
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, "w", zipfile.ZIP_DEFLATED) as zipf:
        for res in selected:
            filepath = os.path.join(UPLOAD_DIR, res['filename'])
            if os.path.exists(filepath):
                zipf.write(filepath, arcname=res['filename'])

    zip_stream.seek(0)

    return StreamingResponse(zip_stream, media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename={zip_filename}"
    })


@app.post("/extract", response_class=HTMLResponse)
async def extract(
    request: Request,
    files: List[UploadFile] = File(...),
    job_description: str = Form(""),
    jd_file: UploadFile = File(None),
    scoring_method: str = Form("gpt")
):
    results = []

    jd_text = ""

    if jd_file is not None and jd_file.filename != "":
        jd_ext = jd_file.filename.lower().split('.')[-1]
        jd_path = os.path.join(UPLOAD_DIR, jd_file.filename)
        with open(jd_path, "wb") as f:
            shutil.copyfileobj(jd_file.file, f)

        if jd_ext == "pdf":
            with fitz.open(jd_path) as doc:
                jd_text = "".join([page.get_text() for page in doc])
        elif jd_ext == "docx":
            jd_text = docx2txt.process(jd_path)

    if not jd_text and job_description.strip():
        jd_text = job_description.strip()

    if not jd_text:
        return templates.TemplateResponse("extract.html", {
            "request": request,
            "error": "Please upload or paste a job description."
        })

    cleaned_jd = clean_resume_text(jd_text)
    jd_exp = extract_years(cleaned_jd)

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        resume_text = extract_text_from_pdf(file_path)
        cleaned_resume = clean_resume_text(resume_text)
        resume_exp = extract_years(cleaned_resume)

        if scoring_method == "gpt":
            match = get_resume_match_score(cleaned_resume, cleaned_jd)

        elif scoring_method == "cosine":
            match = cosine_match_score(cleaned_resume, cleaned_jd)

        elif scoring_method == "hybrid":
            gpt_score = get_resume_match_score(cleaned_resume, cleaned_jd)['score']
            cosine_score = cosine_match_score(cleaned_resume, cleaned_jd)['score']
            keyword_score = keyword_match_score(cleaned_resume, cleaned_jd)['score']

            final_score, exp_reason = enhanced_hybrid_score(
                gpt_score, cosine_score, keyword_score, resume_exp, jd_exp
            )

            match = {
                "score": final_score,
                "reason": f"Hybrid score: GPT={gpt_score}, Cosine={cosine_score}, Keywords={keyword_score}.{exp_reason}"
            }

        else:
            match = {"score": 0, "reason": "Invalid scoring method selected."}

        results.append({
            "filename": file.filename,
            "text": cleaned_resume,
            "score": match['score'],
            "reason": match['reason']
        })
        results.sort(key=lambda x: x['score'], reverse=True)
        LAST_RESULTS.clear()
        LAST_RESULTS.extend(results)

    return templates.TemplateResponse("extract.html", {
        "request": request,
        "results": results,
        "job_description": cleaned_jd,
        "scoring_method": scoring_method
    })
