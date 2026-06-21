import streamlit as st
import pickle

books = pickle.load(open("books.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
matrix = pickle.load(open("matrix.pkl", "rb"))

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Book Recommendation System")
st.caption(
    "Find books similar to your favorites using NLP and Machine Learning"
)

with st.sidebar:

    st.header("About")

    st.write(
        """
        This recommendation engine uses:

        • TF-IDF Vectorization

        • Cosine Similarity

        • Nearest Neighbors
        
        • Streamlit
        """
    )

    st.write(f"Books Available: {len(books):,}")

def recommend(book_name):

    matches = books[
        books["Book-Title"].str.strip().str.lower()
        == book_name.strip().lower()
    ]

    if len(matches) == 0:
        return []

    book_idx = matches.index[0]

    distances, indices = model.kneighbors(
        matrix[book_idx],
        n_neighbors=6
    )

    seen_titles = set()

    recommendations = []

    for idx, i in enumerate(indices[0][1:], start=1):

        title = books.iloc[i]["Book-Title"]

        if title.lower() in seen_titles:
            continue

        seen_titles.add(title.lower())

        similarity_score = round(
        (1 - distances[0][idx]) * 100,
        2
         )

        recommendations.append({
        "Title": title,
        "Author": books.iloc[i]["Book-Author"],
        "Publisher": books.iloc[i]["Publisher"],
        "ISBN": books.iloc[i]["ISBN"],
        "Score": similarity_score
    })

        if len(recommendations) == 5:
            break

    return recommendations

# st.title("📚 Intelligent Book Recommendation System")

selected_book = st.selectbox(
    "Search your favorite book",
    books["Book-Title"].unique(),
    index=None,
    placeholder="Start typing..."
)

if st.button("Get Recommendations"):

    recommendations = recommend(selected_book)

    for book in recommendations:

        with st.container():

            col1, col2 = st.columns([1, 3])

            with col1:

                cover_url = (
                    f"https://covers.openlibrary.org/b/isbn/"
                    f"{book['ISBN']}-L.jpg"
                )

                st.image(cover_url, width=120)

            with col2:

                st.subheader(book["Title"])
                st.write(f"✍️ Author: {book['Author']}")
                st.write(f"🏢 Publisher: {book['Publisher']}")
                st.write(f"🎯 Similarity: {book['Score']}%")

            st.divider()
