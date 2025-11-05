import asyncio
import nest_asyncio
import streamlit as st
from admission_workflow import AdmissionWorkflow, AdmissionStartEvent, default_admission_requirments
import os
import json
from datetime import datetime
import time

# Allow nested event loops in Streamlit
nest_asyncio.apply()

st.set_page_config(
    page_title="Student Admission Management System",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state for applications database
if 'applications' not in st.session_state:
    st.session_state.applications = []

# Data persistence
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "applications_data.json")
DEFAULT_DATA_FILE = os.path.join(SCRIPT_DIR, "applications_data_default.json")

def load_applications():
    """Load applications from default file on startup"""
    if os.path.exists(DEFAULT_DATA_FILE):
        try:
            with open(DEFAULT_DATA_FILE, "r", encoding="utf-8") as f:
                st.session_state.applications = json.load(f)
        except:
            st.session_state.applications = []
    else:
        st.session_state.applications = []

def save_applications():
    """Save applications to file"""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.applications, f, indent=2)

# Load applications on startup - always reload from default
if len(st.session_state.applications) == 0:
    load_applications()

# Sidebar for navigation
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Select Dashboard:",
    ["👨‍🎓 Student Dashboard", "🏫 School Dashboard"],
    label_visibility="collapsed"
)

# ==================== STUDENT DASHBOARD ====================
if page == "👨‍🎓 Student Dashboard":
    st.title("👨‍🎓 Student Application Portal")
    st.markdown("Submit your application materials for admission review")

    st.divider()

    # Student information
    st.subheader("📝 Student Information")
    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Full Name *", value="John Doe")
        student_email = st.text_input("Email *", value="john.doe@email.com")
    with col2:
        student_id = st.text_input("Student ID", value="STU-2024-001")
        student_major = st.text_input("Intended Major", value="Computer Science")

    st.divider()

    # Demo mode toggle
    use_demo_files = st.checkbox("📂 Use Demo Files", value=True, help="Preload sample documents for quick demo")

    # Document uploads
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Required Documents")
        if use_demo_files:
            st.info("✅ Using demo transcript: `transcript.pdf`")
            transcript_file = "demo"
        else:
            transcript_file = st.file_uploader(
                "Upload Transcript (PDF/DOCX)",
                type=["pdf", "docx"],
                help="Student's academic transcript"
            )

        if use_demo_files:
            st.info("✅ Using demo resume: `resume.docx`")
            resume_file = "demo"
        else:
            resume_file = st.file_uploader(
                "Upload Resume (PDF/DOCX)",
                type=["pdf", "docx"],
                help="Student's resume or CV"
            )

    with col2:
        st.subheader("📝 Recommendation Letters")
        if use_demo_files:
            st.info("✅ Using demo letters: `lr1.docx`, `lr2.docx`")
            letter_files = ["demo1", "demo2"]
        else:
            letter_files = st.file_uploader(
                "Upload Recommendation Letters (PDF/DOCX)",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                help="Upload 2 or more recommendation letters"
            )

            if letter_files:
                st.info(f"✅ {len(letter_files)} letter(s) uploaded")

    st.divider()

    # Submit button
    if st.button("🚀 Submit Application", type="primary", use_container_width=True):
        # Validation
        if not student_name or not student_email:
            st.error("❌ Please fill in all required student information")
        elif not transcript_file:
            st.error("❌ Please upload a transcript")
        elif not resume_file:
            st.error("❌ Please upload a resume")
        elif not letter_files or len(letter_files) < 2:
            st.error("❌ Please upload at least 2 recommendation letters")
        else:
            # Save uploaded files
            os.makedirs(os.path.join(SCRIPT_DIR, "temp_uploads"), exist_ok=True)

            application_id = f"APP-{len(st.session_state.applications) + 1:04d}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with st.spinner("📤 Saving uploaded files..."):
                if use_demo_files:
                    # Use demo files from data folder (store as relative paths for portability)
                    transcript_path = "./data/transcript.pdf"
                    resume_path = "./data/resume.docx"
                    letter_paths = ["./data/lr1.docx", "./data/lr2.docx"]
                else:
                    # Use uploaded files (store as relative paths for portability)
                    transcript_path = f"./temp_uploads/{application_id}_transcript_{transcript_file.name}"
                    resume_path = f"./temp_uploads/{application_id}_resume_{resume_file.name}"
                    letter_paths = []

                    # Save files using absolute paths
                    with open(os.path.join(SCRIPT_DIR, transcript_path.lstrip('./')), "wb") as f:
                        f.write(transcript_file.getbuffer())

                    with open(os.path.join(SCRIPT_DIR, resume_path.lstrip('./')), "wb") as f:
                        f.write(resume_file.getbuffer())

                    for i, letter_file in enumerate(letter_files):
                        letter_path = f"./temp_uploads/{application_id}_letter_{i+1}_{letter_file.name}"
                        with open(os.path.join(SCRIPT_DIR, letter_path.lstrip('./')), "wb") as f:
                            f.write(letter_file.getbuffer())
                        letter_paths.append(letter_path)

            # Create application record
            application = {
                "id": application_id,
                "student_name": student_name,
                "student_email": student_email,
                "student_id": student_id if student_id else "N/A",
                "major": student_major if student_major else "Undeclared",
                "submission_time": timestamp,
                "status": "Pending Review",
                "transcript_path": transcript_path,
                "resume_path": resume_path,
                "letter_paths": letter_paths,
                "analysis": None,
                "review": None,
                "processed": False
            }

            st.session_state.applications.append(application)
            save_applications()

            st.success(f"✅ Application submitted successfully! Your application ID is: **{application_id}**")
            st.info("📧 You will be notified via email once your application has been reviewed.")
            st.balloons()

# ==================== SCHOOL DASHBOARD ====================
else:
    st.title("🏫 School Admission Management Dashboard")
    st.markdown("Monitor and process student applications")

    st.divider()

    # Statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Applications", len(st.session_state.applications))
    with col2:
        complete = sum(1 for app in st.session_state.applications if app.get('is_complete', True) and not app['processed'])
        st.metric("✅ Complete", complete)
    with col3:
        incomplete = sum(1 for app in st.session_state.applications if not app.get('is_complete', True))
        st.metric("⚠️ Incomplete", incomplete)
    with col4:
        processed = sum(1 for app in st.session_state.applications if app['processed'])
        st.metric("✔️ Processed", processed)
    with col5:
        if st.button("🔄 Refresh", use_container_width=True):
            load_applications()
            st.rerun()

    st.divider()

    # Admission Requirements Configuration
    admission_requirements = default_admission_requirments
    with st.expander("⚙️ Configure Admission Requirements", expanded=False):
        admission_requirements = st.text_area(
            "Admission Requirements:",
            value=default_admission_requirments,
            height=100,
            help="Edit the admission requirements criteria that will be used to review applications"
        )
        if st.button("💾 Save Requirements"):
            st.success("✅ Requirements saved!")

    st.divider()

    # Applications table
    st.subheader("📋 Application Queue")

    if len(st.session_state.applications) == 0:
        st.info("No applications submitted yet. Waiting for students to submit their applications...")
    else:
        # Two column layout: Queue (left) and Results (right)
        queue_col, result_col = st.columns([1, 1])

        with queue_col:
            st.markdown("### 📋 Current Queue")

            # Filter options
            status_filter = st.selectbox(
                "Filter by Status:",
                ["All", "Pending Review", "Processed"]
            )

            # Filter applications
            filtered_apps = st.session_state.applications
            if status_filter == "Pending Review":
                filtered_apps = [app for app in st.session_state.applications if not app['processed']]
            elif status_filter == "Processed":
                filtered_apps = [app for app in st.session_state.applications if app['processed']]

            st.divider()

            # Display applications in scrollable container
            for i, app in enumerate(filtered_apps):
                is_complete = app.get('is_complete', True)
                missing_docs = app.get('missing_docs', [])

                # Application card
                with st.container():
                    st.markdown(f"**{app['id']}** - {app['student_name']}")
                    st.caption(f"🎓 {app['major']}")

                    # Status badge
                    if is_complete:
                        st.success(f"✅ {app['status']}", icon="✅")
                    else:
                        st.warning(f"⚠️ {app['status']}", icon="⚠️")
                        if missing_docs:
                            st.caption(f"Missing: {', '.join(missing_docs)}")

                    # Action buttons
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if not app['processed']:
                            # Only allow processing for complete applications
                            if is_complete:
                                if st.button(f"🔍 Process", key=f"process_{app['id']}", type="primary", use_container_width=True):
                                    st.session_state['processing_app'] = app['id']
                                    st.rerun()
                            else:
                                st.button(f"❌ Incomplete", key=f"incomplete_{app['id']}", disabled=True, use_container_width=True)
                        else:
                            if st.button(f"📄 View", key=f"view_{app['id']}", type="primary", use_container_width=True):
                                st.session_state['viewing_app'] = app['id']
                                st.rerun()

                    with btn_col2:
                        if app['processed'] and st.session_state.get('viewing_app') == app['id']:
                            if st.button(f"✖️ Close", key=f"close_{app['id']}", use_container_width=True):
                                del st.session_state['viewing_app']
                                st.rerun()

                    st.divider()

        with result_col:
            st.markdown("### 📊 Processing Results")

            # Process application if triggered
            if 'processing_app' in st.session_state:
                app_id = st.session_state['processing_app']
                app = next((a for a in st.session_state.applications if a['id'] == app_id), None)

                if app and not app['processed']:
                    try:
                        st.info(f"**Processing:** {app['student_name']} ({app['id']})")

                        with st.status("🔄 Processing application...", expanded=True) as status:
                            st.write("📋 Loading documents...")
                            st.write("🤖 Analyzing application...")
                            st.write("✍️ Reviewing against admission requirements...")

                            # Create and run workflow
                            async def run_workflow():
                                # Convert relative paths to absolute paths
                                def to_absolute_path(path):
                                    if path and not os.path.isabs(path):
                                        return os.path.join(SCRIPT_DIR, path.lstrip('./'))
                                    return path

                                transcript_abs = to_absolute_path(app['transcript_path'])
                                resume_abs = to_absolute_path(app['resume_path'])
                                letters_abs = [to_absolute_path(p) for p in app['letter_paths']]

                                ad_workflow = AdmissionWorkflow(
                                    admission_requirments=admission_requirements,
                                    timeout=300
                                )
                                ad_startEvent = AdmissionStartEvent(
                                    transcript_file_path=transcript_abs,
                                    resume_file_path=resume_abs,
                                    recommendation_letter_file_path=letters_abs,
                                )
                                return await ad_workflow.run(start_event=ad_startEvent)

                            # Get or create event loop
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                            result = loop.run_until_complete(run_workflow())

                            # Load results
                            analysis = None
                            review = None
                            analysis_path = os.path.join(SCRIPT_DIR, "result", "analysis_result.txt")
                            review_path = os.path.join(SCRIPT_DIR, "result", "review_result.txt")
                            if os.path.exists(analysis_path):
                                with open(analysis_path, "r", encoding="utf-8") as f:
                                    analysis = f.read()
                            if os.path.exists(review_path):
                                with open(review_path, "r", encoding="utf-8") as f:
                                    review = f.read()

                            # Update application
                            app['processed'] = True
                            app['status'] = "Reviewed"
                            app['analysis'] = analysis
                            app['review'] = review
                            save_applications()

                            status.update(label="✅ Processing complete!", state="complete", expanded=False)

                        st.success("🎉 Application processed successfully!")
                        st.session_state['viewing_app'] = app['id']
                        del st.session_state['processing_app']
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error processing application: {str(e)}")
                        if 'processing_app' in st.session_state:
                            del st.session_state['processing_app']

            # View application results if triggered
            elif 'viewing_app' in st.session_state:
                app_id = st.session_state['viewing_app']
                app = next((a for a in st.session_state.applications if a['id'] == app_id), None)

                if app and app['processed']:
                    st.success(f"**Viewing:** {app['student_name']} ({app['id']})")
                    st.caption(f"📧 {app['student_email']} | 🎓 {app['major']}")

                    st.divider()

                    if app['analysis']:
                        st.markdown("#### 📄 Analysis Report")
                        st.text_area("", value=app['analysis'], height=250, key=f"analysis_{app['id']}", label_visibility="collapsed")
                        st.download_button(
                            "📥 Download Analysis",
                            data=app['analysis'],
                            file_name=f"{app['id']}_analysis.txt",
                            mime="text/plain",
                            key=f"dl_analysis_{app['id']}",
                            use_container_width=True
                        )
                        st.divider()

                    if app['review']:
                        st.markdown("#### ✅ Admission Review")
                        st.text_area("", value=app['review'], height=250, key=f"review_{app['id']}", label_visibility="collapsed")
                        st.download_button(
                            "📥 Download Review",
                            data=app['review'],
                            file_name=f"{app['id']}_review.txt",
                            mime="text/plain",
                            key=f"dl_review_{app['id']}",
                            use_container_width=True
                        )
            else:
                st.info("👈 Select an application from the queue to view or process results here.")

# Footer
st.divider()
st.markdown("*Powered by LlamaIndex Workflows & OpenAI*")
