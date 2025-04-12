# Influencer_Match_Maker


Upload and process a dataset of Indian influencers.

Match influencers with campaign briefs using semantic similarity (via sentence-transformers).

Apply filters by audience size, niche, and location.

View top 10 matching influencers with match score, engagement rate, and bios.

Download top matches as a CSV file.


Utilizes the all-MiniLM-L6-v2 model from sentence-transformers to convert influencer bios and the campaign brief into dense embeddings.

Computes similarity using cosine similarity between the campaign brief and influencer bios.

Streamlit is used for fast UI prototyping and interactive web experience.