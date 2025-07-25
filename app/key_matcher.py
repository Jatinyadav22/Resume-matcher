from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def keyword_match_score(resume_text, job_description, top_n=20):
    """
    Compare TF-IDF keywords between resume and job description.
    Returns a score based on overlapping top keywords and explanation.
    """

    # Combine both texts
    documents = [resume_text, job_description]

    # Apply TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Extract TF-IDF scores
    resume_scores = tfidf_matrix[0].toarray().flatten()
    jd_scores = tfidf_matrix[1].toarray().flatten()

    # Get top_n keywords from JD
    top_jd_indices = jd_scores.argsort()[-top_n:][::-1]
    top_jd_keywords = set([feature_names[i] for i in top_jd_indices])

    # Get top_n keywords from Resume
    top_resume_indices = resume_scores.argsort()[-top_n:][::-1]
    top_resume_keywords = set([feature_names[i] for i in top_resume_indices])

    # Calculate intersection
    matched_keywords = top_jd_keywords & top_resume_keywords
    match_ratio = len(matched_keywords) / max(len(top_jd_keywords), 1)
    score = round(match_ratio * 100, 2)

    return {
        "score": score,
        "reason": f"TF-IDF matched {len(matched_keywords)} keywords: {', '.join(matched_keywords)}"
    }
