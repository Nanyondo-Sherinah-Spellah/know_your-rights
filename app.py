
import streamlit as st

from auth import (
    initialize_auth_database,
    register_user,
    authenticate_user,
)

from chat_storage import (
    initialize_database,
    create_conversation,
    list_conversations,
    get_messages,
    save_message,
    rename_conversation,
    delete_conversation,
    set_conversation_title_if_default,
)

from rag import (
    CHAT_MODEL_NAME,
    generate_answer,
    retrieve_passages,
)

from document_ingestion import ingest_pdf

from conversation_export import (
    export_conversation_json,
    export_conversation_csv,
    export_conversation_pdf,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Know Your Rights AI",
    page_icon="⚖️",
    layout="wide",
)


# ==================================================
# DATABASE INITIALIZATION
# ==================================================

initialize_auth_database()
initialize_database()


# ==================================================
# SESSION STATE
# ==================================================

DEFAULT_SESSION_VALUES = {
    "user": None,
    "auth_view": None,
    "current_conversation_id": None,
    "editing_conversation_id": None,
    "deleting_conversation_id": None,
    "selected_model": CHAT_MODEL_NAME,
    "guest_messages": [],
}

for key, value in DEFAULT_SESSION_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_user_session():
    """Log out and return to a fresh guest session."""
    st.session_state.user = None
    st.session_state.auth_view = None
    st.session_state.current_conversation_id = None
    st.session_state.editing_conversation_id = None
    st.session_state.deleting_conversation_id = None
    st.session_state.guest_messages = []

    st.session_state.pop("model_selector", None)


# ==================================================
# AUTHENTICATION
# ==================================================

def show_authentication():
    """Display login or registration inside the sidebar."""

    auth_view = st.session_state.auth_view

    if auth_view == "login":
        st.markdown("### Log in")

        with st.form("login_form"):
            email = st.text_input("Email address")
            password = st.text_input(
                "Password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Log in",
                use_container_width=True,
            )

        if submitted:
            user = authenticate_user(email, password)

            if user is not None:
                st.session_state.user = user
                st.session_state.auth_view = None
                st.session_state.current_conversation_id = None
                st.session_state.editing_conversation_id = None
                st.session_state.deleting_conversation_id = None
                st.session_state.guest_messages = []
                st.rerun()
            else:
                st.error("Invalid email address or password.")

        if st.button(
            "Don't have an account? Sign up",
            key="switch_to_signup",
            use_container_width=True,
        ):
            st.session_state.auth_view = "signup"
            st.rerun()

    elif auth_view == "signup":
        st.markdown("### Create an account")

        with st.form("register_form"):
            email = st.text_input("Email address")
            password = st.text_input(
                "Password (12–128 characters)",
                type="password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Create account",
                use_container_width=True,
            )

        if submitted:
            if password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    user = register_user(email, password)

                    st.session_state.user = user
                    st.session_state.auth_view = None
                    st.session_state.current_conversation_id = None
                    st.session_state.editing_conversation_id = None
                    st.session_state.deleting_conversation_id = None
                    st.session_state.guest_messages = []

                    st.rerun()

                except ValueError as error:
                    st.error(str(error))

                except Exception as error:
                    st.error(
                        "Account creation failed. Please try again."
                    )
                    print(f"Registration error: {error!r}")

        if st.button(
            "Already have an account? Log in",
            key="switch_to_login",
            use_container_width=True,
        ):
            st.session_state.auth_view = "login"
            st.rerun()

    if st.session_state.auth_view is not None:
        if st.button(
            "Back to chat",
            key="back_to_chat",
            use_container_width=True,
        ):
            st.session_state.auth_view = None
            st.rerun()


# The ID is None when the visitor is using the app as a guest.
user_id = (
    st.session_state.user["id"]
    if st.session_state.user is not None
    else None
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .st-key-new_chat_button button {
        background-color: #dc2626 !important;
        color: white !important;
        border: 1px solid #dc2626 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .st-key-new_chat_button button:hover {
        background-color: #b91c1c !important;
        border-color: #b91c1c !important;
        color: white !important;
    }

    [class*="st-key-history-title-"] button {
        background: transparent !important;
        border: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.45rem 0.35rem !important;
        min-height: 2.3rem !important;
        color: inherit !important;
        font-weight: 400 !important;
    }

    [class*="st-key-history-title-"] button:hover {
        background-color: rgba(128, 128, 128, 0.12) !important;
    }

    [class*="st-key-history-edit-"] button,
    [class*="st-key-history-delete-"] button {
        opacity: 0;
        pointer-events: none;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        min-height: 2.3rem !important;
        transition: opacity 0.15s ease-in-out;
    }

    [class*="st-key-history-row-"]:hover
    [class*="st-key-history-edit-"] button,
    [class*="st-key-history-row-"]:hover
    [class*="st-key-history-delete-"] button {
        opacity: 1;
        pointer-events: auto;
    }

    [class*="st-key-history-edit-"] button:hover,
    [class*="st-key-history-delete-"] button:hover {
        background-color: rgba(128, 128, 128, 0.15) !important;
    }

    .history-heading {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-divider {
        margin: 1rem 0;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_OPTIONS = {
    "OpenAI GPT-5 mini": "openai/gpt-5-mini",
    "Anthropic Claude Sonnet 4.5": (
        "anthropic/claude-sonnet-4-5"
    ),
}

MODEL_LABELS = {
    model_id: label
    for label, model_id in MODEL_OPTIONS.items()
}


def get_model_label(model_id):
    """Return a readable model name."""
    if not model_id:
        return "AI assistant"

    return MODEL_LABELS.get(model_id, model_id)


# ==================================================
# CONVERSATION HELPERS
# ==================================================

def start_new_chat():
    """Create a saved conversation or start a guest chat."""

    if user_id is not None:
        conversation_id = create_conversation(user_id)
        st.session_state.current_conversation_id = conversation_id
    else:
        st.session_state.guest_messages = []
        st.session_state.current_conversation_id = None

    st.session_state.editing_conversation_id = None
    st.session_state.deleting_conversation_id = None


def select_conversation(conversation_id):
    """Select a conversation owned by the signed-in user."""

    if user_id is None:
        return

    conversations = list_conversations(user_id)
    owned_ids = {
        conversation["id"]
        for conversation in conversations
    }

    if conversation_id not in owned_ids:
        st.error("Conversation not found.")
        return

    st.session_state.current_conversation_id = conversation_id
    st.session_state.editing_conversation_id = None
    st.session_state.deleting_conversation_id = None


def get_or_create_current_conversation():
    """Get a saved conversation, or use the guest session."""

    if user_id is None:
        return "guest"

    conversations = list_conversations(user_id)
    existing_ids = {
        conversation["id"]
        for conversation in conversations
    }

    conversation_id = st.session_state.current_conversation_id

    if conversation_id not in existing_ids:
        if conversations:
            conversation_id = conversations[0]["id"]
        else:
            conversation_id = create_conversation(user_id)

        st.session_state.current_conversation_id = conversation_id

    return conversation_id


def load_current_messages(conversation_id):
    """Load guest messages or account messages."""

    if user_id is None:
        return st.session_state.guest_messages

    return get_messages(user_id, conversation_id)


def save_current_message(
    conversation_id,
    role,
    content,
    model=None,
    sources=None,
):
    """Save to the correct storage based on authentication."""

    if user_id is not None:
        save_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            sources=sources or [],
        )
    else:
        st.session_state.guest_messages.append(
            {
                "role": role,
                "content": content,
                "model": model,
                "sources": sources or [],
            }
        )


def get_export_conversation(conversation_id, messages):
    """Build export metadata and messages for the current chat."""

    if user_id is not None:
        conversations = list_conversations(user_id)

        conversation = next(
            (
                item
                for item in conversations
                if item["id"] == conversation_id
            ),
            None,
        )

        if conversation is None:
            raise ValueError("Conversation not found.")

        # Recheck ownership through the storage layer.
        export_messages = get_messages(user_id, conversation_id)

        return conversation, export_messages

    # Guest conversations are held in the current session.
    return {
        "id": None,
        "title": "Guest conversation",
        "created_at": None,
        "updated_at": None,
    }, messages


# ==================================================
# SOURCE DISPLAY
# ==================================================

def display_sources(sources):
    """Display retrieved passages in an expandable section."""

    if not sources:
        return

    with st.expander(
        f"📚 Sources ({len(sources)})",
        expanded=False,
    ):
        for index, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                article = (
                    source.get("article")
                    or source.get("title")
                    or source.get("chapter")
                    or source.get("source_document")
                    or "Knowledge-base passage"
                )

                source_text = (
                    source.get("text")
                    or source.get("content")
                    or ""
                )

                st.markdown(f"**Source {index}: {article}**")

                source_document = source.get("source_document")

                if source_document:
                    st.caption(f"Document: {source_document}")

                if source_text:
                    st.write(source_text)
            else:
                st.write(source)

            if index < len(sources):
                st.markdown("---")


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.title("⚖️ Know Your Rights AI")

    st.caption(
        "Your guide to understanding the Constitution of Uganda."
    )

    # ----------------------------------------------
    # SIGNED-IN USER CONTROLS
    # ----------------------------------------------

    if user_id is not None:
        st.caption(
            f"Signed in as {st.session_state.user['email']}"
        )

        if st.button(
            "Log out",
            key="logout_button",
            use_container_width=True,
        ):
            clear_user_session()
            st.rerun()

    st.markdown("---")

    # ----------------------------------------------
    # NEW CHAT
    # ----------------------------------------------

    with st.container(key="new_chat_button"):
        if st.button(
            "＋ New chat",
            key="new_chat",
            use_container_width=True,
        ):
            start_new_chat()
            st.rerun()

    # ----------------------------------------------
    # CONVERSATION HISTORY
    # ----------------------------------------------

    st.markdown(
        '<div class="history-heading">'
        "Your conversations</div>",
        unsafe_allow_html=True,
    )

    conversations = (
        list_conversations(user_id)
        if user_id is not None
        else []
    )

    if user_id is not None and not conversations:
        st.caption("Your saved conversations will appear here.")

    for conversation in conversations:
        conversation_id_item = conversation["id"]
        conversation_title = (
            conversation["title"] or "New conversation"
        )

        with st.container(
            key=f"history-row-{conversation_id_item}"
        ):
            title_col, edit_col, delete_col = st.columns(
                [8, 1, 1],
                vertical_alignment="center",
                gap="small",
            )

            with title_col:
                if st.button(
                    conversation_title,
                    key=f"history-title-{conversation_id_item}",
                    use_container_width=True,
                    help=conversation_title,
                ):
                    select_conversation(conversation_id_item)
                    st.rerun()

            with edit_col:
                if st.button(
                    "✏️",
                    key=f"history-edit-{conversation_id_item}",
                    help="Rename conversation",
                    use_container_width=True,
                ):
                    st.session_state.editing_conversation_id = (
                        conversation_id_item
                    )
                    st.session_state.deleting_conversation_id = None
                    st.rerun()

            with delete_col:
                if st.button(
                    "🗑️",
                    key=f"history-delete-{conversation_id_item}",
                    help="Delete conversation",
                    use_container_width=True,
                ):
                    st.session_state.deleting_conversation_id = (
                        conversation_id_item
                    )
                    st.session_state.editing_conversation_id = None
                    st.rerun()

        # ------------------------------------------
        # RENAME FORM
        # ------------------------------------------

        if (
            st.session_state.editing_conversation_id
            == conversation_id_item
        ):
            with st.container(border=True):
                st.caption("Rename conversation")

                with st.form(
                    key=f"rename-form-{conversation_id_item}"
                ):
                    new_title = st.text_input(
                        "Conversation title",
                        value=conversation_title,
                        max_chars=100,
                        label_visibility="collapsed",
                    )

                    save_col, cancel_col = st.columns(2)

                    with save_col:
                        save_clicked = st.form_submit_button(
                            "Save",
                            use_container_width=True,
                        )

                    with cancel_col:
                        cancel_clicked = st.form_submit_button(
                            "Cancel",
                            use_container_width=True,
                        )

                    if save_clicked:
                        cleaned_title = new_title.strip()

                        if cleaned_title:
                            try:
                                rename_conversation(
                                    user_id,
                                    conversation_id_item,
                                    cleaned_title,
                                )

                                st.session_state.editing_conversation_id = None
                                st.rerun()

                            except ValueError:
                                st.error("Conversation not found.")
                        else:
                            st.warning(
                                "Please enter a conversation title."
                            )

                    if cancel_clicked:
                        st.session_state.editing_conversation_id = None
                        st.rerun()

        # ------------------------------------------
        # DELETE CONFIRMATION
        # ------------------------------------------

        if (
            st.session_state.deleting_conversation_id
            == conversation_id_item
        ):
            with st.container(border=True):
                st.warning(
                    f'Delete "{conversation_title}"? '
                    "This will permanently remove its saved messages."
                )

                confirm_col, cancel_col = st.columns(2)

                with confirm_col:
                    if st.button(
                        "Delete",
                        key=f"confirm-delete-{conversation_id_item}",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            delete_conversation(
                                user_id,
                                conversation_id_item,
                            )

                            st.session_state.deleting_conversation_id = None
                            st.session_state.editing_conversation_id = None

                            if (
                                st.session_state.current_conversation_id
                                == conversation_id_item
                            ):
                                remaining = list_conversations(user_id)

                                st.session_state.current_conversation_id = (
                                    remaining[0]["id"]
                                    if remaining
                                    else None
                                )

                            st.rerun()

                        except ValueError:
                            st.error("Conversation not found.")

                with cancel_col:
                    if st.button(
                        "Cancel",
                        key=f"cancel-delete-{conversation_id_item}",
                        use_container_width=True,
                    ):
                        st.session_state.deleting_conversation_id = None
                        st.rerun()

    # ----------------------------------------------
    # MODEL SELECTION
    # ----------------------------------------------

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    st.subheader("AI model")

    model_labels = list(MODEL_OPTIONS.keys())
    model_ids = list(MODEL_OPTIONS.values())

    current_model = st.session_state.selected_model

    if current_model not in model_ids:
        current_model = (
            CHAT_MODEL_NAME
            if CHAT_MODEL_NAME in model_ids
            else model_ids[0]
        )
        st.session_state.selected_model = current_model

    selected_label = MODEL_LABELS.get(
        current_model,
        model_labels[0],
    )

    selected_label = st.selectbox(
        "Choose a model",
        options=model_labels,
        index=model_labels.index(selected_label),
        key="model_selector",
        label_visibility="collapsed",
    )

    st.session_state.selected_model = MODEL_OPTIONS[selected_label]

    # ----------------------------------------------
    # PDF KNOWLEDGE-BASE UPLOAD
    # ----------------------------------------------

    st.markdown("---")
    st.subheader("Knowledge base")

    st.caption(
        "Add PDF documents to expand the information available "
        "to the assistant. Uploaded documents are not automatically "
        "official or authoritative legal sources."
    )

    uploaded_pdf = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        accept_multiple_files=False,
        max_upload_size=20,
        key="knowledge_base_pdf_uploader",
        help="PDF files only, up to 20 MB and 300 pages.",
    )

    if uploaded_pdf is not None:
        st.write(f"**Selected:** {uploaded_pdf.name}")

        st.caption(
            f"File size: "
            f"{uploaded_pdf.size / (1024 * 1024):.2f} MB"
        )

        if st.button(
            "Add to knowledge base",
            key="add_knowledge_base_pdf",
            use_container_width=True,
            type="primary",
        ):
            pdf_bytes = uploaded_pdf.getvalue()

            if not pdf_bytes.startswith(b"%PDF-"):
                st.error(
                    "This file does not appear to be a valid PDF."
                )

            elif len(pdf_bytes) > 20 * 1024 * 1024:
                st.error(
                    "PDF files must be 20 MB or smaller."
                )

            else:
                try:
                    with st.spinner(
                        "Extracting PDF text and creating embeddings..."
                    ):
                        result = ingest_pdf(
                            pdf_bytes=pdf_bytes,
                            filename=uploaded_pdf.name,
                        )

                    if result["status"] == "already_indexed":
                        st.info(result["message"])
                    else:
                        st.success(
                            f"{result['message']} "
                            f"Chunks indexed: {result['chunks']}."
                        )

                    st.caption(
                        "The original Constitution records are preserved."
                    )

                except ValueError as error:
                    st.error(
                        f"Could not process this PDF: {error}"
                    )

                except Exception as error:
                    st.error(
                        "The document could not be indexed. "
                        "Check the terminal for technical details."
                    )
                    print(f"PDF ingestion error: {error!r}")

    # ----------------------------------------------
    # LOGIN BUTTON — BOTTOM OF SIDEBAR
    # ----------------------------------------------

    st.markdown("---")

    if user_id is None:
        if st.session_state.auth_view is not None:
            show_authentication()
        else:
            if st.button(
                "Log in",
                key="show_login",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.auth_view = "login"
                st.rerun()

    # ----------------------------------------------
    # LEGAL DISCLAIMER
    # ----------------------------------------------

    st.markdown("---")

    st.caption(
        "Know Your Rights AI helps you explore the Constitution "
        "of Uganda. Verify important legal matters against "
        "authoritative sources."
    )


# ==================================================
# MAIN CHAT AREA
# ==================================================

st.title("Know Your Rights AI")

st.markdown(
    "Ask questions about the Constitution of Uganda and explore "
    "the constitutional provisions behind the answers."
)

st.info(
    "This tool provides constitutional information for educational "
    "purposes. It is not a substitute for professional legal advice."
)


# ==================================================
# LOAD SELECTED CONVERSATION
# ==================================================

conversation_id = get_or_create_current_conversation()
messages = load_current_messages(conversation_id)


# ==================================================
# CONVERSATION EXPORT
# ==================================================

with st.expander("📥 Export this conversation", expanded=False):
    if messages:
        try:
            export_conversation, export_messages = (
                get_export_conversation(conversation_id, messages)
            )

            st.caption(
                "Download the currently open conversation. "
                "Your messages and available source references are included."
            )

            pdf_data = export_conversation_pdf(
                export_conversation,
                export_messages,
            )

            csv_data = export_conversation_csv(export_messages)

            json_data = export_conversation_json(
                export_conversation,
                export_messages,
            )

            pdf_col, csv_col, json_col = st.columns(3)

            with pdf_col:
                st.download_button(
                    label="📄 PDF",
                    data=pdf_data,
                    file_name="know_your_rights_conversation.pdf",
                    mime="application/pdf",
                    key="export_pdf",
                    use_container_width=True,
                )

            with csv_col:
                st.download_button(
                    label="📊 CSV",
                    data=csv_data.encode("utf-8-sig"),
                    file_name="know_your_rights_conversation.csv",
                    mime="text/csv",
                    key="export_csv",
                    use_container_width=True,
                )

            with json_col:
                st.download_button(
                    label="{} JSON",
                    data=json_data.encode("utf-8"),
                    file_name="know_your_rights_conversation.json",
                    mime="application/json",
                    key="export_json",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(
                "Unable to prepare the conversation exports. "
                "Please try again."
            )
            print(f"Conversation export error: {error!r}")

    else:
        st.caption(
            "Start a conversation to enable downloads."
        )


# ==================================================
# DISPLAY CONVERSATION MESSAGES
# ==================================================

for message in messages:
    role = message["role"]
    content = message["content"]

    if role not in ("user", "assistant"):
        continue

    with st.chat_message(role):
        if role == "assistant":
            model_id = message.get("model")

            if model_id:
                st.caption(
                    f"Answered using {get_model_label(model_id)}"
                )

        st.markdown(content)

        if role == "assistant":
            display_sources(message.get("sources") or [])


# ==================================================
# CHAT INPUT AND RESPONSE GENERATION
# ==================================================

question = st.chat_input(
    "Ask a question about the Constitution of Uganda..."
)

if question:
    question = question.strip()

    if question:
        model_for_question = st.session_state.selected_model

        # ------------------------------------------
        # SAVE USER QUESTION
        # ------------------------------------------

        save_current_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
            model=model_for_question,
            sources=[],
        )

        # Set a title only for saved account conversations.
        if user_id is not None:
            set_conversation_title_if_default(
                user_id,
                conversation_id,
                question,
            )

        with st.chat_message("user"):
            st.markdown(question)

        # ------------------------------------------
        # GENERATE ASSISTANT RESPONSE
        # ------------------------------------------

        with st.chat_message("assistant"):
            st.caption(
                f"Answered using "
                f"{get_model_label(model_for_question)}"
            )

            with st.spinner(
                "Searching the knowledge base and preparing an answer..."
            ):
                try:
                    passages = retrieve_passages(question)

                    result = generate_answer(
                        question,
                        passages,
                        model=model_for_question,
                    )

                    if isinstance(result, dict):
                        answer = (
                            result.get("answer")
                            or result.get("content")
                            or (
                                "I couldn't generate an answer. "
                                "Please try again."
                            )
                        )

                        sources = (
                            result.get("sources")
                            or result.get("passages")
                            or passages
                        )
                    else:
                        answer = str(result)
                        sources = passages

                    st.markdown(answer)
                    display_sources(sources)

                    save_current_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=answer,
                        model=model_for_question,
                        sources=sources,
                    )

                except Exception as error:
                    error_message = (
                        "I couldn't generate an answer because "
                        "the request failed. Please try again."
                    )

                    st.error(error_message)

                    save_current_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=error_message,
                        model=model_for_question,
                        sources=[],
                    )

                    print(f"Answer generation error: {error!r}")

        st.rerun()
