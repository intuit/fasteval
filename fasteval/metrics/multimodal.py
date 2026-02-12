"""Multi-modal evaluation metrics.

This module provides metrics for evaluating multi-modal AI systems
that combine text, images, audio, and other modalities.

Includes:
- Multi-modal RAG metrics
- Cross-modal coherence
- Table/document extraction

Requires: pip install fasteval[multimodal] or specific extras
"""

import logging
from typing import Any, List, Optional

from fasteval.metrics.llm import BaseLLMMetric
from fasteval.metrics.vision import BaseVisionMetric
from fasteval.models.evaluation import EvalInput, MetricResult
from fasteval.providers.base import LLMClient

logger = logging.getLogger(__name__)


class MultiModalFaithfulnessMetric(BaseVisionMetric):
    """
    Evaluate if response is faithful to multi-modal context.

    Similar to text faithfulness, but evaluates grounding across
    text documents, images, and other context types.

    Example:
        @fe.multimodal_faithfulness(threshold=0.8)
        def test_document_qa():
            response = rag_system.query(
                question="What was Q3 revenue?",
                documents=["financial_report.pdf"]
            )
            fe.score(
                actual_output=response,
                context=retrieved_text_chunks,
                images=[ImageInput(source="chart_page_5.png")],
                expected="$4.2M"
            )
    """

    def __init__(
        self,
        name: str = "multimodal_faithfulness",
        threshold: float = 0.8,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        # Format text context if available
        text_context = ""
        if eval_input.context or eval_input.retrieval_context:
            ctx = eval_input.context or eval_input.retrieval_context or []
            text_context = "\n\n**Text Context Documents**:\n" + "\n".join(
                f"[{i+1}] {doc}" for i, doc in enumerate(ctx)
            )

        # Note about images
        image_note = ""
        images = eval_input.get_all_images()
        if images:
            image_note = f"\n\n**Visual Context**: [{len(images)} image(s) provided - examine carefully]"

        return f"""You are an expert evaluator assessing faithfulness in multi-modal RAG systems.

## TASK
Evaluate whether the RESPONSE is FAITHFUL to the provided MULTI-MODAL CONTEXT
(both text documents and images). All claims must be grounded in the context.

## INPUTS
**Query**: {eval_input.input}
**Response to Evaluate**: {eval_input.actual_output}{text_context}{image_note}

## MULTI-MODAL FAITHFULNESS METHODOLOGY

### Step 1: Claim Extraction
Extract ALL factual claims from the response:
- Textual claims (facts, figures, statements)
- Visual claims (descriptions, data from charts/images)
- Cross-modal claims (integrating text and visual information)

### Step 2: Source Attribution
For EACH claim, determine its source and support:
- **TEXT_SUPPORTED**: Claim verified by text context documents
- **IMAGE_SUPPORTED**: Claim verified by visual context (images)
- **CROSS_MODAL_SUPPORTED**: Claim requires both text and image to verify
- **NOT_SUPPORTED**: Claim cannot be verified from any provided context
- **CONTRADICTED**: Claim conflicts with the context

### Step 3: Cross-Modal Verification
Check for claims that require integration of multiple sources:
- Does the response correctly synthesize text and visual information?
- Are there inconsistencies between modalities that are ignored?
- Is cross-modal reasoning accurate?

## FAITHFULNESS CALCULATION
Score = (Supported Claims) / (Total Claims)
- TEXT_SUPPORTED, IMAGE_SUPPORTED, CROSS_MODAL_SUPPORTED = 1.0
- NOT_SUPPORTED = 0.0
- CONTRADICTED = 0.0 (also flag as critical)

## SCORING RUBRIC
- **1.0**: Completely faithful - all claims supported by text or images
- **0.8**: Highly faithful - nearly all claims supported, minor unsupported details
- **0.6**: Mostly faithful - majority supported but some unsupported claims
- **0.4**: Partially faithful - significant unsupported content
- **0.2**: Mostly unfaithful - majority of claims not grounded in context
- **0.0**: Unfaithful - claims contradict context or entirely fabricated

## OUTPUT FORMAT
Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<list each claim with source (TEXT/IMAGE/CROSS_MODAL/NOT_SUPPORTED/CONTRADICTED), calculate faithfulness score>"}}"""


class TableExtractionMetric(BaseVisionMetric):
    """
    Evaluate accuracy of table data extraction from images.

    Specifically designed for evaluating extraction of tabular data
    from document images, PDFs, or screenshots.

    Example:
        @fe.table_extraction(threshold=0.9)
        def test_table_qa():
            response = model.answer(
                image="financial_table.png",
                question="What is the total for 2024?"
            )
            fe.score(
                actual_output=response,
                expected="$150,000",
                image="financial_table.png"
            )
    """

    def __init__(
        self,
        name: str = "table_extraction",
        threshold: float = 0.9,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        expected_str = ""
        if eval_input.expected_output:
            expected_str = f"\nExpected Answer: {eval_input.expected_output}"

        fields_str = ""
        if eval_input.expected_fields:
            fields_str = "\nExpected Fields:\n" + "\n".join(
                f"  - {k}: {v}" for k, v in eval_input.expected_fields.items()
            )

        return f"""Evaluate the accuracy of table data extraction from the image.

Query: {eval_input.input}
Extracted Response: {eval_input.actual_output}{expected_str}{fields_str}

Instructions:
1. Check if the correct cells/values were identified in the table
2. Verify numbers are extracted accurately (including decimals, currency)
3. Check if row/column headers are interpreted correctly
4. Verify any calculations or aggregations are correct

Provide your evaluation as JSON:
{{"score": <0.0 to 1.0>, "reasoning": "<describe extraction accuracy>"}}

Score 0.0 = Completely incorrect extraction
Score 0.5 = Some values correct but significant errors
Score 1.0 = Perfect extraction matching expected values"""


class FigureReferenceMetric(BaseVisionMetric):
    """
    Evaluate if response correctly references figures and charts.

    Checks if references to visual elements in documents are accurate
    and properly attributed.

    Example:
        @fe.figure_reference(threshold=0.8)
        def test_figure_citation():
            response = model.analyze(
                images=["doc_with_figures.png"],
                query="Summarize the trends shown in the figures"
            )
            fe.score(
                actual_output=response,
                images=["doc_with_figures.png"]
            )
    """

    def __init__(
        self,
        name: str = "figure_reference",
        threshold: float = 0.8,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        return f"""Evaluate if the response correctly references figures, charts, and visual elements.

Query: {eval_input.input}
Response: {eval_input.actual_output}

Instructions:
1. Identify all references to visual elements in the response
2. Check if each reference accurately describes what's in the images
3. Verify figure numbers/labels are correctly attributed
4. Check if data cited from figures is accurate

Provide your evaluation as JSON:
{{"score": <0.0 to 1.0>, "reasoning": "<describe reference accuracy>"}}

Score 0.0 = References are incorrect or misattributed
Score 0.5 = Some references correct but errors present
Score 1.0 = All figure references are accurate"""


class CrossModalCoherenceMetric(BaseVisionMetric):
    """
    Evaluate coherence between text and visual elements.

    Checks if generated text is consistent with provided images
    and vice versa - useful for captioning, description generation,
    and multi-modal generation tasks.

    Example:
        @fe.cross_modal_coherence(threshold=0.8)
        def test_caption_coherence():
            caption = model.caption(image="photo.jpg")
            fe.score(
                actual_output=caption,
                image="photo.jpg"
            )
    """

    def __init__(
        self,
        name: str = "cross_modal_coherence",
        threshold: float = 0.8,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        return f"""You are an expert evaluator assessing cross-modal coherence between text and images.

## TASK
Evaluate the COHERENCE between the generated TEXT and the visual content in the IMAGE(s).
Check if text and visuals are aligned, consistent, and mutually supportive.

## INPUTS
**Generated Text**: {eval_input.actual_output}
**Visual Context**: [Image(s) provided - examine carefully]

## CROSS-MODAL COHERENCE DIMENSIONS

### 1. Semantic Alignment
- Does the text content match what's shown in the image?
- Are the same concepts, objects, and scenes described?
- Is the meaning consistent across modalities?

### 2. Factual Consistency
- Do specific details (numbers, colors, counts) match?
- Are there any factual contradictions between text and image?
- Do descriptions accurately reflect visual elements?

### 3. Entity Correspondence
- Are entities mentioned in text visible in the image?
- Are visual elements appropriately described in text?
- Is there a clear mapping between textual and visual entities?

### 4. Stylistic Coherence
- Is the tone of the text appropriate for the visual content?
- Does the text style match the image style/mood?
- Is there tonal consistency across modalities?

### 5. Completeness of Coverage
- Does the text adequately describe key visual elements?
- Are important visual details represented in text?
- Is anything significant in the image omitted from text?

## EVALUATION METHODOLOGY
Step 1: Identify key visual elements in the image(s)
Step 2: Identify key claims/descriptions in the text
Step 3: Map text descriptions to visual elements
Step 4: Check for mismatches, contradictions, or omissions
Step 5: Assess overall cross-modal alignment

## COHERENCE ISSUES TO DETECT
- Text describes objects not present in image
- Image contains elements not mentioned in text
- Color, size, or attribute mismatches
- Positional or spatial inconsistencies
- Tone/mood mismatch between modalities

## SCORING RUBRIC
- **1.0**: Perfect coherence - text and image fully aligned, mutually supportive
- **0.8**: High coherence - strong alignment with minor omissions
- **0.6**: Moderate coherence - generally aligned but some inconsistencies
- **0.4**: Partial coherence - noticeable misalignments or gaps
- **0.2**: Low coherence - significant contradictions or missing connections
- **0.0**: No coherence - text and image are unrelated or contradictory

## OUTPUT FORMAT
Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<map text elements to visual elements, identify alignments and mismatches, assess overall coherence>"}}"""


class DocumentUnderstandingMetric(BaseVisionMetric):
    """
    Evaluate understanding of complex documents with mixed content.

    Designed for document AI tasks that involve understanding
    documents containing text, tables, figures, and forms.

    Example:
        @fe.document_understanding(threshold=0.8)
        def test_doc_qa():
            response = model.query(
                document="contract.pdf",
                question="What is the termination clause?"
            )
            fe.score(
                actual_output=response,
                expected_output="30-day written notice",
                image="contract.pdf"  # or images for each page
            )
    """

    def __init__(
        self,
        name: str = "document_understanding",
        threshold: float = 0.8,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        expected_str = ""
        if eval_input.expected_output:
            expected_str = (
                f"\n**Expected Answer** (ground truth): {eval_input.expected_output}"
            )

        return f"""You are an expert evaluator assessing document AI understanding capabilities.

## TASK
Evaluate whether the RESPONSE demonstrates accurate UNDERSTANDING of the DOCUMENT
and correctly answers the query based on document content.

## INPUTS
**Query**: {eval_input.input}
**Model Response**: {eval_input.actual_output}{expected_str}

## DOCUMENT UNDERSTANDING DIMENSIONS

### 1. Content Extraction Accuracy
- Are text passages correctly read and extracted?
- Are numbers, dates, names accurately captured?
- Is key information from the document present?

### 2. Layout Understanding
- Is document structure (headers, sections, paragraphs) understood?
- Are multi-column layouts handled correctly?
- Is reading order respected?

### 3. Table Comprehension
- Are table structures correctly interpreted?
- Are row/column relationships understood?
- Are cell values accurately extracted?

### 4. Figure/Chart Integration
- Are charts and graphs correctly interpreted?
- Are figure captions associated with correct figures?
- Is visual data accurately reported?

### 5. Form Field Recognition
- Are form labels and values correctly associated?
- Are checkbox/selection states understood?
- Are structured fields accurately captured?

### 6. Cross-Section Reasoning
- Is information synthesized from multiple document sections?
- Are relationships between document parts understood?
- Is contextual information from the document used appropriately?

## EVALUATION STEPS
Step 1: Identify document elements relevant to the query
Step 2: Extract what the response claims about document content
Step 3: Verify accuracy of extracted information against document
Step 4: Check if document structure was understood correctly
Step 5: Assess if the query is fully and accurately answered

## COMMON DOCUMENT UNDERSTANDING ERRORS
- Misreading OCR errors or poor quality text
- Confusing table rows/columns
- Missing information from specific sections
- Incorrect figure references
- Ignoring footnotes or fine print

## SCORING RUBRIC
- **1.0**: Perfect understanding - all document elements correctly interpreted, query fully answered
- **0.8**: Strong understanding - minor errors, query well answered
- **0.6**: Moderate understanding - some extraction errors, query partially answered
- **0.4**: Weak understanding - significant errors, incomplete answer
- **0.2**: Poor understanding - major misinterpretations, barely addresses query
- **0.0**: No understanding - completely wrong or fails to use document

## OUTPUT FORMAT
Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<assess each dimension, verify document extraction accuracy, check query coverage>"}}"""


class CLIPScoreMetric(BaseLLMMetric):
    """
    Evaluate image-text alignment using CLIP-style scoring.

    While the actual CLIP model requires PyTorch (available with image-gen extra),
    this metric uses an LLM to simulate CLIP-style alignment scoring.

    For production image generation evaluation with actual CLIP scores,
    install: pip install fasteval[image-gen]

    Example:
        @fe.clip_score(threshold=0.7)
        def test_image_alignment():
            image = dalle.generate("A red sports car on a mountain road")
            fe.score(
                generated_image=image,
                input="A red sports car on a mountain road"
            )
    """

    def __init__(
        self,
        name: str = "clip_score",
        threshold: float = 0.7,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        """Note: For actual CLIP scores, use the image-gen extra with PyTorch."""
        return f"""Evaluate the semantic alignment between an image and its text description.

Text Description/Prompt: {eval_input.input}

Simulate a CLIP-style alignment score by considering:
1. How well does the image match the text description?
2. Are all key elements from the text present in the image?
3. Is the style/mood consistent with what's described?
4. Would a retrieval system match this text to this image?

Provide your evaluation as JSON:
{{"score": <0.0 to 1.0>, "reasoning": "<describe alignment analysis>"}}

Score 0.0 = No alignment between text and (hypothetical) image
Score 0.5 = Partial alignment, some elements match
Score 1.0 = Perfect alignment, text accurately describes image"""


class AestheticScoreMetric(BaseVisionMetric):
    """
    Evaluate the aesthetic quality of generated images.

    Assesses visual appeal, composition, and artistic merit.

    Example:
        @fe.aesthetic_score(threshold=0.7)
        def test_image_aesthetics():
            image = model.generate("A beautiful sunset")
            fe.score(generated_image=image, input="A beautiful sunset")
    """

    def __init__(
        self,
        name: str = "aesthetic_score",
        threshold: float = 0.7,
        weight: float = 1.0,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            threshold=threshold,
            weight=weight,
            llm_client=llm_client,
            model=model,
            **kwargs,
        )

    def _prepare_images(self, eval_input: EvalInput) -> List[str]:
        """Prepare generated image for evaluation."""
        from fasteval.utils.image import PILLOW_AVAILABLE, prepare_images_for_api

        if not PILLOW_AVAILABLE:
            raise ImportError(
                "Vision metrics require the 'vision' extra. "
                "Install with: pip install fasteval[vision]"
            )

        if eval_input.generated_image:
            from fasteval.models.multimodal import GeneratedImage, ImageInput

            if isinstance(eval_input.generated_image, GeneratedImage):
                img = eval_input.generated_image.image
            elif isinstance(eval_input.generated_image, ImageInput):
                img = eval_input.generated_image
            else:
                img = eval_input.generated_image

            return prepare_images_for_api(
                [img],
                max_dimension=self.max_image_dimension,
            )

        return super()._prepare_images(eval_input)

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        return f"""You are an expert art critic and visual designer evaluating image aesthetics.

## TASK
Evaluate the AESTHETIC QUALITY of the image based on visual design principles.

## CONTEXT
**Generation Prompt/Context**: {eval_input.input or "N/A"}

## AESTHETIC DIMENSIONS (Evaluate Each)

### 1. Composition (25% weight)
- **Balance**: Is the visual weight distributed appropriately?
- **Rule of Thirds**: Are key elements placed at strong points?
- **Leading Lines**: Do lines guide the eye effectively?
- **Focal Point**: Is there a clear subject or point of interest?
- **Negative Space**: Is empty space used effectively?

### 2. Color & Tone (25% weight)
- **Color Harmony**: Do colors work well together?
- **Contrast**: Is there appropriate tonal range?
- **Mood**: Do colors convey an appropriate emotional tone?
- **Saturation**: Are color intensities balanced?
- **Color Temperature**: Is warm/cool balance appropriate?

### 3. Lighting (20% weight)
- **Light Quality**: Hard vs soft, natural vs artificial
- **Light Direction**: Does lighting enhance the subject?
- **Shadows**: Are shadows used effectively?
- **Exposure**: Is brightness appropriate, no blown highlights/crushed blacks?
- **Atmosphere**: Does lighting create depth and mood?

### 4. Technical Quality (15% weight)
- **Sharpness**: Is focus appropriate where needed?
- **Noise/Grain**: Is noise level acceptable?
- **Artifacts**: Are there generation artifacts, banding, or defects?
- **Resolution**: Is detail level sufficient?
- **Distortion**: Are there unwanted distortions?

### 5. Artistic Merit (15% weight)
- **Creativity**: Is there originality or unique perspective?
- **Emotional Impact**: Does the image evoke a response?
- **Style Coherence**: Is there a consistent artistic style?
- **Visual Interest**: Does the image engage the viewer?
- **Storytelling**: Does the image convey a narrative or meaning?

## EVALUATION APPROACH
Step 1: Assess each dimension independently (1-10)
Step 2: Note specific strengths and weaknesses
Step 3: Calculate weighted overall aesthetic score
Step 4: Consider context - is the aesthetic appropriate for the intended use?

## SCORING RUBRIC
- **1.0**: Exceptional aesthetics - professional quality, visually stunning
- **0.8**: High aesthetics - very appealing, minor imperfections
- **0.6**: Good aesthetics - pleasant, meets expectations
- **0.4**: Fair aesthetics - acceptable but unremarkable
- **0.2**: Poor aesthetics - noticeable flaws, unappealing
- **0.0**: Very poor aesthetics - major flaws, visually unpleasant

## OUTPUT FORMAT
Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<assess each dimension with specific observations, calculate weighted score>"}}"""
