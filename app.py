import streamlit as st
import os
import logging
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directories to H: drive
os.environ.setdefault("HF_HOME", "H:/AI Cognitive Partner/ai-engineering-hub-main/shared-cache/huggingface")
os.environ.setdefault("TORCH_HOME", "H:/AI Cognitive Partner/ai-engineering-hub-main/shared-cache/torch")
os.environ.setdefault("TRANSFORMERS_CACHE", "H:/AI Cognitive Partner/ai-engineering-hub-main/shared-cache/transformers")
os.environ.setdefault("FASTEMBED_CACHE", "H:/AI Cognitive Partner/ai-engineering-hub-main/shared-cache/fastembed")

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("ACP_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
TEMP_DIR = PROJECT_ROOT / "temp"
LOGS_DIR = PROJECT_ROOT / "logs"
BACKUPS_DIR = PROJECT_ROOT / "backups"

for d in [DATA_DIR, MODELS_DIR, CACHE_DIR, TEMP_DIR, LOGS_DIR, BACKUPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AI Cognitive Partner V2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design system (light theme)
from ui.components import inject_design_system  # noqa: E402
inject_design_system()


# ---------------------------------------------------------------------------
# Demo mode path resolution
# ---------------------------------------------------------------------------

def get_active_db_path() -> str:
    """Return the active SQLite database path based on demo mode."""
    if st.session_state.get("acp_demo_mode", False):
        from demo_db_manager import DEMO_DB_PATH
        return str(DEMO_DB_PATH)
    return str(DATA_DIR / "local_memory.db")


def get_active_faiss_path() -> str:
    """Return the active FAISS index path based on demo mode."""
    if st.session_state.get("acp_demo_mode", False):
        from demo_db_manager import DEMO_FAISS_PATH
        return str(DEMO_FAISS_PATH)
    return str(DATA_DIR / "faiss_index.bin")


def get_active_data_dir() -> str:
    """Return the active data directory based on demo mode."""
    if st.session_state.get("acp_demo_mode", False):
        from demo_db_manager import DEMO_DATA_DIR
        return str(DEMO_DATA_DIR)
    return str(DATA_DIR)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "selected_sources" not in st.session_state:
        st.session_state.selected_sources = set()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "pipeline_initialized" not in st.session_state:
        st.session_state.pipeline_initialized = False
    if "processed_hashes" not in st.session_state:
        st.session_state.processed_hashes = set()
    if "selected_course" not in st.session_state:
        st.session_state.selected_course = None
    if "current_video" not in st.session_state:
        st.session_state.current_video = None
    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Pipeline initialization (backend — supports demo mode)
# ---------------------------------------------------------------------------
def initialize_pipeline():
    if st.session_state.pipeline_initialized:
        return True
    try:
        with st.spinner("Initializing AI Cognitive Partner V2..."):
            from src.document_processing.unified_processor import UnifiedDocumentProcessor
            from src.embeddings.embedding_generator import EmbeddingGenerator
            from src.courses.course_material_service import CourseMaterialService
            from src.vector_database.faiss_vector_db import FAISSVectorStore
            from src.llm.local_llm import LocalLLM
            from src.generation.rag import RAGGenerator
            from src.memory.memory_layer import NotebookMemoryLayer

            doc_processor = UnifiedDocumentProcessor()
            embedding_generator = EmbeddingGenerator()

            # Use demo paths if in demo mode, otherwise production paths
            index_path = get_active_faiss_path()
            meta_db_path = get_active_db_path()
            data_dir = get_active_data_dir()

            vector_db = FAISSVectorStore(
                index_path=index_path,
                meta_db_path=meta_db_path,
                embedding_dim=embedding_generator.get_embedding_dimension(),
            )

            # Ensure a default course exists for backward compatibility
            if not st.session_state.get("acp_demo_mode", False):
                vector_db.ensure_course("legacy-default", title="Legacy Default Course")

            course_material_service = CourseMaterialService(
                vector_store=vector_db,
                embedding_generator=embedding_generator,
                data_dir=data_dir,
                document_processor=doc_processor,
            )

            llm = LocalLLM()
            rag_generator = RAGGenerator(
                embedding_generator=embedding_generator,
                vector_db=vector_db,
                llm=llm,
            )

            # Get authenticated user id for memory layer
            auth_user = st.session_state.get("acp_auth_user") or {}
            user_id = auth_user.get("id", "v2_user")

            memory = NotebookMemoryLayer(
                db_path=meta_db_path,
                user_id=user_id,
                session_id=st.session_state.session_id,
                create_new_session=False,
            )

            st.session_state.pipeline = {
                "doc_processor": doc_processor,
                "embedding_generator": embedding_generator,
                "vector_db": vector_db,
                "course_material_service": course_material_service,
                "llm": llm,
                "rag_generator": rag_generator,
                "memory": memory,
                "db_path": meta_db_path,
                "faiss_path": index_path,
                "data_dir": data_dir,
            }
            st.session_state.pipeline_initialized = True
            return True
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        logger.error(f"Init error: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# File processing (backend — supports demo mode)
# ---------------------------------------------------------------------------
def process_uploaded_files(uploaded_files):
    if not st.session_state.pipeline:
        return
    from ui.navigation import current_role
    if current_role() not in ("teacher", "administrator"):
        st.error("Only teachers and administrators can upload course materials.")
        return
    service = st.session_state.pipeline["course_material_service"]
    course_id = st.session_state.get("selected_course")
    if not course_id:
        st.error("Select a course before uploading materials.")
        return
    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
        for uploaded_file in uploaded_files:
            try:
                result = service.import_pdf(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    course_id,
                )
                if result["status"] == "duplicate_other_course":
                    st.warning(result["message"])
                elif result["status"] == "duplicate":
                    st.info(result["message"])
                else:
                    st.success(result["message"])
            except Exception as error:
                st.error(f"Failed to process {uploaded_file.name}: {error}")
                logger.error(
                    "Material processing failed for %s: %s",
                    uploaded_file.name,
                    error,
                    exc_info=True,
                )


def remove_source(name, doc_id=None):
    pipeline = st.session_state.get("pipeline")
    if pipeline:
        vdb = pipeline.get("vector_db")
        if vdb:
            try:
                if doc_id:
                    doc = next((d for d in vdb.list_documents() if d.get("doc_id") == doc_id), None)
                    if doc and doc.get("filename") == name:
                        vdb.delete_document(doc_id)
                    else:
                        for doc in vdb.list_documents():
                            if doc.get("filename") == name:
                                vdb.delete_document(doc["doc_id"])
                else:
                    for doc in vdb.list_documents():
                        if doc.get("filename") == name:
                            vdb.delete_document(doc["doc_id"])
            except Exception as e:
                logger.error(f"Vector deletion error for {name}: {e}", exc_info=True)
        mem = pipeline.get("memory")
        if mem:
            try:
                mem.delete_document_metadata(name)
            except Exception as e:
                logger.error(f"Memory metadata deletion error for {name}: {e}", exc_info=True)

    for src in st.session_state.sources:
        if src.get("name") == name and src.get("hash"):
            st.session_state.processed_hashes.discard(src["hash"])

    st.session_state.sources = [s for s in st.session_state.sources if not (s.get("name") == name and (doc_id is None or s.get("doc_id") == doc_id))]
    st.session_state.selected_sources.discard(name)


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------
def _render_page(page: str) -> None:
    """Dispatch to the registered page module."""
    from ui.pages import (
        welcome,
        teacher_dashboard,
        course_management,
        assignments_evaluations,
        students_analytics,
        student_dashboard,
        my_learning,
        student_submissions,
        ai_workspace,
        admin_dashboard,
        admin_users,
        admin_courses,
        admin_system,
    )

    registry = {
        "welcome": welcome.render,
        "teacher_dashboard": teacher_dashboard.render,
        "course_management": course_management.render,
        "assignments_evaluations": assignments_evaluations.render,
        "students_analytics": students_analytics.render,
        "student_dashboard": student_dashboard.render,
        "my_learning": my_learning.render,
        "student_submissions": student_submissions.render,
        "ai_workspace": ai_workspace.render,
        "admin_dashboard": admin_dashboard.render,
        "admin_users": admin_users.render,
        "admin_courses": admin_courses.render,
        "admin_system": admin_system.render,
    }
    renderer = registry.get(page, welcome.render)
    if page in {"course_management", "my_learning"}:
        renderer(pipeline=st.session_state.get("pipeline"))
    else:
        renderer()


def main():
    # Backend session state
    init_session_state()

    # UI session state (role, page, user)
    from ui.navigation import init_ui_state, is_authenticated, current_page, is_demo_mode
    init_ui_state()

    # Route
    if not is_authenticated():
        from ui.pages.welcome import render as welcome_render
        welcome_render()
        return

    page = current_page()

    # Lazy-initialize the backend pipeline only when entering pages that need it.
    if page in {"ai_workspace", "course_management", "my_learning"}:
        if not initialize_pipeline():
            st.stop()

    # Role shell (topbar + sidebar + content)
    from ui.layout import layout_shell
    from ui.navigation import current_role

    if page == "ai_workspace":
        from ui.pages.ai_workspace import render as workspace_render
        workspace_render(
            pipeline=st.session_state.get("pipeline"),
            process_files_fn=process_uploaded_files,
            remove_source_fn=remove_source,
        )
        return

    layout_shell(current_role(), lambda: _render_page(page))


if __name__ == "__main__":
    main()
