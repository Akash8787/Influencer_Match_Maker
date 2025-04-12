import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_PATH = "indian_influencers.csv"  # Adjust this path if needed
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Audience Size'] = df['Audience Size'].astype(int)

    # Add dummy Engagement Rate if not present
    if 'Engagement Rate' not in df.columns:
        np.random.seed(42)  # For reproducible results
        df['Engagement Rate'] = np.random.uniform(1.0, 5.0, len(df)).round(2)
    return df

def main():
    st.set_page_config(page_title="Creator Matchmaker", layout="wide")
    st.title("🎯 Creator Matchmaker Pro")
    st.caption("Find the perfect creators for your campaign using AI-powered matching")
    
    df = load_data()
    model = load_model()
    
    if 'embeddings' not in st.session_state:
        with st.spinner("Initializing creator database..."):
            st.session_state.embeddings = model.encode(df['Bio'].tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        campaign_brief = st.text_area(
            "**Paste your campaign brief:**",
            height=200,
            placeholder="Example: We're looking for tech creators who can review our new smartphone with camera-focused features..."
        )
    with col2:
        st.write("**Filters**")
        st.write("Audience Size (Followers)")
        min_followers = st.number_input("Minimum (K)", min_value=1, max_value=10000, value=20, step=1)
        max_followers = st.number_input("Maximum (K)", min_value=1, max_value=10000, value=5000, step=1)
        
        niche_filter = st.multiselect(
            "Niche (optional)",
            options=sorted(df['Niche'].unique()),
            help="Select one or more niches to narrow down creators."
        )
        
        location_filter = st.multiselect(
            "Location (optional)",
            options=sorted(df['Location'].unique()),
            help="Select one or more locations to filter creators."
        )
    
    if st.button("Find Top Creators", type="primary"):
        if not campaign_brief.strip():
            st.error("Please enter a campaign brief")
            return
        
        if min_followers > max_followers:
            st.error("Minimum followers cannot exceed maximum followers")
            return
        
        with st.spinner("Analyzing creators..."):
            filtered_df = df[
                (df['Audience Size'] >= min_followers * 1000) &
                (df['Audience Size'] <= max_followers * 1000)
            ]
            if niche_filter:
                filtered_df = filtered_df[filtered_df['Niche'].isin(niche_filter)]
            if location_filter:
                filtered_df = filtered_df[filtered_df['Location'].isin(location_filter)]
            
            if filtered_df.empty:
                st.warning("No creators match your filters. Try adjusting your criteria.")
                return
            
            filtered_indices = filtered_df.index.tolist()
            filtered_embeddings = st.session_state.embeddings[filtered_indices]
            brief_embedding = model.encode([campaign_brief])
            similarities = cosine_similarity(brief_embedding, filtered_embeddings)
            filtered_df['Match Score'] = (similarities[0] * 100).round(1)
            results = filtered_df.sort_values('Match Score', ascending=False).head(10)
        
        st.subheader(f"Top {len(results)} Matches")
        filter_summary = f"Showing creators with {min_followers}K–{max_followers}K followers"
        if niche_filter:
            filter_summary += f" in {', '.join(niche_filter)} niches"
        if location_filter:
            filter_summary += f" located in {', '.join(location_filter)}"
        st.caption(filter_summary)
        
        csv = results[['Name', 'Niche', 'Location', 'Audience Size', 'Bio', 'Match Score']].to_csv(index=False)
        st.download_button(
            label="Download Top Matches as CSV",
            data=csv,
            file_name="top_creators.csv",
            mime="text/csv",
        )
        
        cols = st.columns(2)
        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"#### {row['Name']} ({row['Niche']})")
                    st.caption(f"📍 {row['Location']} | 👥 {row['Audience Size']/1000:.1f}K | ⭐ {row['Engagement Rate']}% ER")
                    st.progress(row['Match Score']/100, text=f"**{row['Match Score']}% Match**")
                    st.write(row['Bio'])
                    st.button("Contact", key=f"contact_{idx}", use_container_width=True)

if __name__ == "__main__":
    main()
