
import streamlit as st

from rag import retrieve_passages, generate_answer


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Know Your Rights AI",
    page_icon="⚖️",
    layout="centered",
)

# --------------------------------------------------
# 2. Page header
# --------------------------------------------------

st.title("⚖️ Know Your Rights AI")

st.markdown(
    """
    Understand your rights through Uganda's Constitution.

    Ask a question in plain language. The assistant will retrieve
    relevant constitutional provisions and use them to formulate
    an answer.
    """
)

st.info(
    "This tool provides constitutional information for educational "
    "purposes. It is not a substitute for advice from a qualified "
    "legal professional."
)

st.divider()

# --------------------------------------------------
# 3. Initialise conversation state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 4. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources attached to assistant responses.
        if message["role"] == "assistant":
            passages = message.get("sources", [])

            if passages:
                with st.expander("View constitutional sources"):
                    for index, passage in enumerate(passages, start=1):
                        article = passage.get("article_number")
                        title = passage.get("title")
                        section_type = passage.get("section_type")

                        if article:
                            source_title = f"Article {article}"
                            if title and str(title) != str(article):
                                source_title += f" — {title}"
                        else:
                            source_title = title or section_type or (
                                f"Source {index}"
                            )

                        st.markdown(f"**{source_title}**")

                        if passage.get("chapter"):
                            st.caption(
                                f"Chapter: {passage['chapter']}"
                            )

                        st.write(passage.get("text", ""))

                        if index < len(passages):
                            st.divider()


# --------------------------------------------------
# 5. Handle a new question
# --------------------------------------------------

question = st.chat_input(
    "Ask about your constitutional rights..."
)

if question:
    # Display and save the user's question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve constitutional passages and generate an answer.
    with st.chat_message("assistant"):
        with st.spinner(
            "Searching the Constitution and preparing your answer..."
        ):
            try:
                passages = retrieve_passages(question)

                if not passages:
                    answer = (
                        "I could not find relevant constitutional "
                        "passages for that question. Try rephrasing "
                        "it or asking about a specific constitutional right."
                    )
                    sources = []
                else:
                    answer = generate_answer(question, passages)
                    sources = passages

                st.markdown(answer)

                if sources:
                    with st.expander("View constitutional sources"):
                        for index, passage in enumerate(
                            sources, start=1
                        ):
                            article = passage.get("article_number")
                            title = passage.get("title")
                            section_type = passage.get("section_type")

                            if article:
                                source_title = f"Article {article}"
                                if title and str(title) != str(article):
                                    source_title += f" — {title}"
                            else:
                                source_title = (
                                    title
                                    or section_type
                                    or f"Source {index}"
                                )

                            st.markdown(f"**{source_title}**")

                            if passage.get("chapter"):
                                st.caption(
                                    f"Chapter: {passage['chapter']}"
                                )

                            st.write(passage.get("text", ""))

                            if index < len(sources):
                                st.divider()

                # Save the answer and its sources for future reruns.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception:
                st.error(
                    "Something went wrong while answering your question. "
                    "Check your terminal for the error details, then "
                    "try again."
                )
                raise


# --------------------------------------------------
# 6. Sidebar controls
# --------------------------------------------------

with st.sidebar:
    st.subheader("About this assistant")

    st.write(
        "Know Your Rights AI uses retrieval-augmented generation "
        "(RAG) to find relevant passages from its constitutional "
        "knowledge base before generating answers."
    )

    st.caption(
        "Answers depend on the content and quality of the indexed "
        "constitutional documents."
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
