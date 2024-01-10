import textdistance as td


def match(resume_TFIDF, job_des_TFIDF , resume_cleaned, job_des_cleaned):
    j = td.jaccard.similarity(resume_cleaned, job_des_cleaned)
    s = td.sorensen_dice.similarity(resume_cleaned, job_des_cleaned)
    c = td.cosine.similarity(resume_TFIDF, job_des_TFIDF)
    o = td.overlap.normalized_similarity(resume_TFIDF, job_des_TFIDF)
    total = (j+s+c+o)/4
    return total*100
