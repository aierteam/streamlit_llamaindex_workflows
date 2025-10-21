import asyncio
import nest_asyncio
import streamlit as st
from admission_workflow import AdmissionWorkflow, AdmissionStartEvent, default_admission_requirments
import os

# Allow nested event loops in Streamlit
nest_asyncio.apply()

st.set_page_config(
    page_title="Student Admission Review System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Admission Review System")
st.markdown("Upload student application materials for automated analysis and review.")

# Admission Requirements Section
st.subheader("📋 Admission Requirements")
admission_requirements = st.text_area(
    "Customize admission requirements:",
    value=default_admission_requirments,
    height=100,
    help="Edit the admission requirements criteria that will be used to review applications"
)

st.divider()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Required Documents")
    transcript_file = st.file_uploader(
        "Upload Transcript (PDF/DOCX)",
        type=["pdf", "docx"],
        help="Student's academic transcript"
    )
    resume_file = st.file_uploader(
        "Upload Resume (PDF/DOCX)",
        type=["pdf", "docx"],
        help="Student's resume or CV"
    )

with col2:
    st.subheader("📝 Recommendation Letters")
    letter_files = st.file_uploader(
        "Upload Recommendation Letters (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload 2 or more recommendation letters"
    )

    if letter_files:
        st.info(f"✅ {len(letter_files)} letter(s) uploaded")

# Process button
st.divider()

if st.button("🚀 Process Application", type="primary", use_container_width=True):
    # Validation
    if not transcript_file:
        st.error("❌ Please upload a transcript")
    elif not resume_file:
        st.error("❌ Please upload a resume")
    elif not letter_files or len(letter_files) < 2:
        st.error("❌ Please upload at least 2 recommendation letters")
    else:
        # Save uploaded files temporarily
        os.makedirs("./temp_uploads", exist_ok=True)

        with st.spinner("📤 Saving uploaded files..."):
            transcript_path = f"./temp_uploads/{transcript_file.name}"
            resume_path = f"./temp_uploads/{resume_file.name}"
            letter_paths = []

            with open(transcript_path, "wb") as f:
                f.write(transcript_file.getbuffer())

            with open(resume_path, "wb") as f:
                f.write(resume_file.getbuffer())

            for i, letter_file in enumerate(letter_files):
                letter_path = f"./temp_uploads/{letter_file.name}"
                with open(letter_path, "wb") as f:
                    f.write(letter_file.getbuffer())
                letter_paths.append(letter_path)

        # Run the workflow
        try:
            with st.status("🔄 Processing application...", expanded=True) as status:
                st.write("📋 Uploading documents...")
                st.write("🤖 Analyzing application...")
                st.write("✍️ Reviewing against admission requirements...")

                # Create and run workflow
                async def run_workflow():
                    ad_workflow = AdmissionWorkflow(
                        admission_requirments=admission_requirements,
                        timeout=300
                    )
                    ad_startEvent = AdmissionStartEvent(
                        transcript_file_path=transcript_path,
                        resume_file_path=resume_path,
                        recommendation_letter_file_path=letter_paths,
                    )
                    return await ad_workflow.run(start_event=ad_startEvent)

                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                result = loop.run_until_complete(run_workflow())

                status.update(label="✅ Processing complete!", state="complete", expanded=False)

            # Display results
            st.success("🎉 Application processing completed successfully!")

            # Show results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Full Result", "📄 Analysis", "✅ Review"])

            with tab1:
                st.markdown("### Complete Result")
                st.text_area("", value=str(result), height=400, key="full_result")

            with tab2:
                if os.path.exists("./analysis_result.txt"):
                    with open("./analysis_result.txt", "r", encoding="utf-8") as f:
                        analysis = f.read()
                    st.markdown("### Application Analysis")
                    st.text_area("", value=analysis, height=400, key="analysis")
                    st.download_button(
                        "📥 Download Analysis",
                        data=analysis,
                        file_name="analysis_result.txt",
                        mime="text/plain"
                    )

            with tab3:
                if os.path.exists("./review_result.txt"):
                    with open("./review_result.txt", "r", encoding="utf-8") as f:
                        review = f.read()
                    st.markdown("### Admission Review")
                    st.text_area("", value=review, height=400, key="review")
                    st.download_button(
                        "📥 Download Review",
                        data=review,
                        file_name="review_result.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"❌ Error processing application: {str(e)}")
            st.exception(e)

# Footer
st.divider()
st.markdown("*Powered by LlamaIndex Workflows & OpenAI*")
