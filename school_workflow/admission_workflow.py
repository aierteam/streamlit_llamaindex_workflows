import logging
import time
from typing import List
from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.workflow import (
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI

from llama_index.utils.workflow import draw_all_possible_flows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("admission_workflow.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


analyze_application_template = PromptTemplate(
    """You are an admissions officer reviewing student applications.
    Analyze the following student application materials:

    Transcript: {transcript}

    Resume: {resume}

    Recommendation Letters: {recommendation_letters}

    Provide a summary of the strengths and weaknesses of the application.
    """
)

default_admission_requirments = """- GPA >= 3.0"""
admission_requirement_template = PromptTemplate(
    """You are an admissions officer.   
    Review the following application materials and determine if they meet the admission requirements:
        {admission_requirments}
    
    Transcript: {transcript}

    Resume: {resume}

    Recommendation Letters: {recommendation_letters}

    """
)


class AdmissionStartEvent(StartEvent):
    transcript_file_path: str | None = None
    resume_file_path: str | None = None
    recommendation_letter_file_path: List[str] | None = None


class AnalyzeApplicationEvent(Event):
    pass


class ReviewApplicationEvent(Event):
    pass


class AdmissionWorkflow(Workflow):
    def __init__(
        self,
        admission_requirments,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.llm = OpenAI(model="gpt-5", service_tier="priority")
        self.admission_requirments = admission_requirments
        self.transcript = None
        self.resume = None
        self.recommendation_letters = []

    # @step
    # async def upload_transcript_step(
    #     self, ctx: Context, ev: AdmissionStartEvent
    # ) -> AnalyzeApplicationEvent:
    #     if ev.transcript_file_path:
    #         self.upload_transcript(ev.transcript_file_path)
    #         return AnalyzeApplicationEvent()

    # @step
    # async def upload_resume_step(
    #     self, ctx: Context, ev: AdmissionStartEvent
    # ) -> AnalyzeApplicationEvent:
    #     if ev.resume_file_path:
    #         self.upload_resume(ev.resume_file_path)
    #         return AnalyzeApplicationEvent()

    # @step
    # async def upload_recommendation_letter_step(
    #     self, ctx: Context, ev: AdmissionStartEvent
    # ) -> AnalyzeApplicationEvent:
    #     if ev.recommendation_letter_file_path:
    #         for letter_path in ev.recommendation_letter_file_path:
    #             self.upload_recommendation_letter(letter_path)
    #         return AnalyzeApplicationEvent()

    @step
    async def upload_documents_step(
        self, ctx: Context, ev: AdmissionStartEvent
    ) -> AnalyzeApplicationEvent:
        if ev.transcript_file_path:
            self.upload_transcript(ev.transcript_file_path)
        if ev.resume_file_path:
            self.upload_resume(ev.resume_file_path)
        if ev.recommendation_letter_file_path:
            for letter_path in ev.recommendation_letter_file_path:
                self.upload_recommendation_letter(letter_path)
        return AnalyzeApplicationEvent()

    @step
    async def analyze_application_step(
        self, ctx: Context, ev: AnalyzeApplicationEvent
    ) -> ReviewApplicationEvent | StopEvent:
        analysis = self.analyze_application()
        await ctx.store.set("analysis", analysis)
        with open("./analysis_result.txt", "w", encoding="utf-8") as f:
            f.write(str(analysis))
        if self.is_complete():
            return ReviewApplicationEvent()
        else:
            return StopEvent(result=analysis)

    @step
    async def review_application_step(
        self, ctx: Context, ev: ReviewApplicationEvent
    ) -> StopEvent:
        review = self.reivew_application()
        with open("./review_result.txt", "w", encoding="utf-8") as f:
            f.write(str(review))
        analysis = await ctx.store.get("analysis")
        final_result = f"Analysis:\n{analysis}\n\nReview:\n{review}"
        return StopEvent(result=final_result)

    def is_complete(self) -> bool:
        is_complete = (
            self.transcript is not None
            and self.resume is not None
            and len(self.recommendation_letters) >= 2
        )
        return is_complete

    def upload_transcript(self, transcript_path: str) -> None:
        """
        Upload and store the student's transcript.

        Args:
            transcript_path: Path to the transcript file
        """
        reader = SimpleDirectoryReader(input_files=[transcript_path])
        docs = reader.load_data()
        self.transcript = docs[0] if docs else None

    def upload_resume(self, resume_path: str) -> None:
        """
        Upload and store the student's resume.

        Args:
            resume_path: Path to the resume file
        """
        reader = SimpleDirectoryReader(input_files=[resume_path])
        docs = reader.load_data()
        self.resume = docs[0] if docs else None

    def upload_recommendation_letter(self, letter_path: str) -> None:
        """
        Upload and store a recommendation letter.

        Args:
            letter_path: Path to the recommendation letter file
        """
        reader = SimpleDirectoryReader(input_files=[letter_path])
        docs = reader.load_data()
        if docs:
            self.recommendation_letters.append(docs[0])

    def analyze_application(self) -> str:
        """
        Analyze the student's application including transcript, resume, and recommendation letters.

        Returns:
            A string summarizing the analysis of the application
        """
        transcript_text = (
            self.transcript.text if self.transcript else "No transcript provided."
        )
        resume_text = self.resume.text if self.resume else "No resume provided."
        recommendation_letters_text = (
            "\n".join([letter.text for letter in self.recommendation_letters if letter])
            if self.recommendation_letters != []
            else "No recommendation letters provided."
        )
        analysis_prompt = analyze_application_template.format(
            transcript=transcript_text,
            resume=resume_text,
            recommendation_letters=recommendation_letters_text,
        )

        start_time = time.time()
        analysis = self.llm.complete(analysis_prompt)
        end_time = time.time()
        logger.info(f"LLM analysis completion took {end_time - start_time:.2f} seconds")
        return analysis

    def reivew_application(self) -> str:
        """
        Review the student's application and provide feedback.

        Returns:
            A string summarizing the review of the application
        """
        transcript_text = (
            self.transcript.text if self.transcript else "No transcript provided."
        )
        resume_text = self.resume.text if self.resume else "No resume provided."
        recommendation_letters_text = (
            "\n".join([letter.text for letter in self.recommendation_letters if letter])
            if self.recommendation_letters != []
            else "No recommendation letters provided."
        )
        admission_requirments = self.admission_requirments
        review_prompt = admission_requirement_template.format(
            admission_requirments=admission_requirments,
            transcript=transcript_text,
            resume=resume_text,
            recommendation_letters=recommendation_letters_text,
        )

        start_time = time.time()
        review = self.llm.complete(review_prompt)
        end_time = time.time()
        logger.info(f"LLM review completion took {end_time - start_time:.2f} seconds")
        return review


import asyncio


async def main():
    ad_workflow = AdmissionWorkflow(
        admission_requirments=default_admission_requirments, timeout=300
    )
    ad_startEvent = AdmissionStartEvent(
        transcript_file_path="./data/transcript.pdf",
        resume_file_path="./data/resume.docx",
        recommendation_letter_file_path=["./data/lr1.docx", "./data/lr2.docx"],
    )
    result = await ad_workflow.run(start_event=ad_startEvent)
    print(result)


if __name__ == "__main__":
    draw_all_possible_flows(AdmissionWorkflow)
    asyncio.run(main())
