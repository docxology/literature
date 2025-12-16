"""Research task templates for LLM operations."""
from __future__ import annotations

from typing import Optional

from infrastructure.llm.templates.base import ResearchTemplate


class SummarizeAbstract(ResearchTemplate):
    """Template for summarizing abstracts."""
    template_str = (
        "Please summarize the following abstract in 3-5 bullet points, "
        "highlighting the main contribution, methodology, and results:\n\n"
        "${text}"
    )


class LiteratureReview(ResearchTemplate):
    """Template for generating literature reviews."""
    template_str = (
        "Based on the following paper summaries, write a cohesive "
        "literature review paragraph:\n\n"
        "${summaries}"
    )


class CodeDocumentation(ResearchTemplate):
    """Template for documenting code."""
    template_str = (
        "Generate a Python docstring for the following code, "
        "including Args, Returns, and Raises sections:\n\n"
        "${code}"
    )


class DataInterpretation(ResearchTemplate):
    """Template for interpreting data."""
    template_str = (
        "Analyze the following data statistics and provide "
        "a scientific interpretation of the trends:\n\n"
        "${stats}"
    )


class PaperSummarization(ResearchTemplate):
    """Template for comprehensive paper summarization.

    Generates a detailed summary of a research paper focusing on
    relevance, comprehensiveness, and specificity rather than rigid structure.
    Emphasizes extracting key information accurately and substantively.
    Supports domain-specific variants for better context-aware summarization.
    """
    template_str = """=== PAPER CONTENT ===

Title: ${title}
Authors: ${authors}
Year: ${year}
Source: ${source}
${domain_info}

PAPER TEXT:
${text}

=== END PAPER CONTENT ===

CRITICAL ACCURACY VERIFICATION (MUST DO FIRST):
Before writing your summary, you MUST complete these verification steps:

1. Read the paper title above: "${title}"
2. Read the abstract and introduction sections of the paper text carefully
3. Verify: Is this paper actually about the topic indicated in the title?
4. Extract 5-10 key terms from the title and abstract (e.g., if title mentions "Freudian Psychoanalysis", look for terms like "psychoanalysis", "Freud", "defense mechanisms", "conflict")
5. If you cannot verify the paper is about the stated topic, or if the paper text contradicts the title, STOP and report an error
6. Only proceed with summarization if you can confirm the paper matches the title

TITLE MATCHING REQUIREMENT:
- Your summary MUST begin by correctly stating the paper title
- The title in your summary MUST exactly match: "${title}"
- If the paper title mentions specific terms (e.g., "Freudian Psychoanalysis", "Deep Neural Networks", "Active Inference"), 
  your summary MUST discuss these exact terms and concepts
- If your summary discusses topics NOT mentioned in the title (e.g., discussing "neural networks" when title says "psychoanalysis"), 
  this indicates a CRITICAL ERROR - you are summarizing the wrong paper

EVIDENCE REQUIREMENTS:
- Include at least 3-5 direct quotes or closely paraphrased quotes from the paper
   - QUOTE EXTRACTION AND FORMATTING:
     * Extract quotes VERBATIM from the paper text - do NOT modify or "correct" them
     * Extract quotes exactly as they appear in the source text
     * Preserve all aspects of the quote exactly as written, including spacing
     * Use proper quotation marks: "quote text" (double quotes)
     * CRITICAL: Only extract quotes that actually appear in the paper text - do NOT generate or invent quotes
   - QUOTE FORMATTING STANDARD:
  * Attribution format: "The authors state: \"quote text\"" OR "According to the paper: \"quote text\""
  * Vary attribution phrases to avoid repetition (use: "The authors state", "They note", 
    "The paper argues", "According to the research", "The study demonstrates")
  * Include section context when available: "In the Introduction, the authors state: \"quote text\""
  * Ensure proper spacing around quotes and punctuation
- Format quotes as: "quote text" or use brackets for paraphrases: [paraphrased claim from paper]
- Every major claim must reference specific text from the paper
- Include section/page references when possible (e.g., "As stated in the Introduction: ..." or "The abstract notes: ...")
- When discussing key findings, cite the specific text: "The paper states: [quote]" or "According to the results section: [paraphrase]"
- Avoid making claims without evidence - if you cannot find supporting text in the paper, do not include that claim

TOPIC VALIDATION:
- Extract 5-10 key terms from the paper title and abstract
- Ensure your summary discusses these key terms explicitly
- If the summary doesn't mention key terms from the title, this indicates a major error
- Verify that the topics discussed in your summary align with the topics in the paper text
- If you find yourself writing about topics not present in the paper (e.g., writing about "adversarial attacks" when the paper is about "psychoanalysis"), 
  STOP - you are hallucinating content

PROFESSIONAL TONE REQUIREMENTS:
- Begin directly with the paper title or content - NO conversational openings
- Do NOT use phrases like: "Okay, here's...", "Here's a summary...", 
  "Let me summarize...", "I'll extract...", or similar conversational language
- Start immediately with substantive content in formal academic tone
- Example BAD: "Okay, here's a summary of the paper..."
- Example GOOD: "This paper investigates [topic]..."

CRITICAL INSTRUCTIONS:
You are summarizing a scientific research paper. You MUST follow ALL rules below:

1. ONLY use information that appears in the paper text above. Do NOT add external knowledge, assumptions, or invented details.

2. Provide a comprehensive summary that covers the key aspects of the paper. Use section headers that make sense for the content, such as:
   - Overview/Summary (what the paper is about)
   - Key Contributions/Findings (main results and advances)
   - Methodology/Approach (how the research was conducted)
   - Results/Data (what was found or measured)
   - Limitations/Discussion (weaknesses and future work)

3. Word count: Aim for 600-1000 words of comprehensive, technically detailed content. Prioritize comprehensiveness and technical depth over brevity. Longer, more detailed summaries are preferred.

4. CRITICAL ANTI-REPETITION RULES - STRICTLY ENFORCED (ENHANCED):
   - Each sentence must be UNIQUE - check before writing if you've already said this
   - Each claim appears ONCE - if you've stated it, move to the next unique point
   - Each section must contain UNIQUE information. Do NOT repeat the same content across different sections.
   - Do NOT create multiple sections with the same title (e.g., multiple "Summary" sections).
   - Each paragraph must be UNIQUE. Do NOT repeat the same paragraph, even with slight variations.
   - Each sentence must be UNIQUE. Do NOT repeat the same sentence multiple times.
   - Do NOT repeat the same claim, finding, or explanation multiple times in different words.
   - Vary attribution phrases: use "The authors state", "They note", "The paper argues", 
     "According to the research", "The study demonstrates" - do NOT repeat the same phrase
   - If you find yourself writing similar content, STOP and write something completely different.
   - Use each section header only ONCE. Never create duplicate sections.
   - If you notice you're repeating yourself, delete the repetition and write new unique content instead.
   - CRITICAL: Before writing each sentence, ask: "Have I already said this?" If yes, write something new.

5. CONTENT FOCUS - EXTRACT SPECIFIC CLAIMS AND TECHNICAL DETAILS:
   - Extract SPECIFIC claims: What exactly does the paper claim to show or prove? Be concrete, not generic.
   - Include NUMERICAL VALUES: performance metrics (%, accuracy, F1-score, precision, recall), sample sizes (n=), 
     statistical tests (p-values, effect sizes, confidence intervals), experimental parameters, hyperparameters
   - Include EQUATIONS: mathematical formulations, key equations with notation explained, mathematical objects
   - Document SPECIFIC FINDINGS: concrete results with numbers (e.g., "94.2% accuracy", "p < 0.001", "n=32 participants", 
     "3.5x speedup", "reduced error by 23%"), not generic descriptions
   - Technical terminology: Use exact terms from paper (algorithm names, brain regions, mathematical objects, 
     dataset names, model architectures)
   - Experimental details: dataset sizes, number of participants, experimental conditions, hardware specifications,
     training parameters, evaluation metrics
   - Explain what makes this paper UNIQUE: How does it differ from or improve upon related work? What is novel about 
     the approach or findings? Include specific comparisons when available
   - Methodology specifics: Document what was measured and how (exact procedures, tools, techniques)
   - Avoid generic descriptions: Instead of "the paper presents methods", say "the paper presents a novel algorithm 
     that achieves 15.3% improvement on ImageNet dataset with ResNet-50 backbone"

6. DOMAIN-SPECIFIC EMPHASIS:
${domain_instructions}

7. QUALITY STANDARDS:
   - Be substantive: Provide detailed analysis rather than surface-level descriptions
   - Explain significance: Discuss why methods, results, and contributions matter
   - Maintain coherence: Ensure different sections complement rather than repeat each other
   - Use evidence: Support claims with specific details from the paper
   - One section, one purpose: Each section should have a distinct purpose and unique content

8. ACCURACY REQUIREMENTS:
   - NO HALLUCINATION: Only discuss what the paper explicitly states
   - NO REPETITION WHATSOEVER: 
     * Do NOT repeat the same sentence, even once
     * Do NOT repeat the same paragraph, even with slight variations
     * Do NOT repeat the same claim or finding multiple times
     * If you catch yourself repeating, DELETE the repetition and write something new
   - NO SECTION DUPLICATION: Each section title should appear only once - never create duplicate sections
   - NO PARAGRAPH REPETITION: Every paragraph must be completely unique
   - NO SENTENCE REPETITION: Every sentence must be completely unique
   - NO META-COMMENTARY: Do not mention being an AI or that this is a summary
   - SCIENTIFIC TONE: Use formal, academic language throughout

9. FLEXIBLE STRUCTURE: Use the section headers that best fit the paper's content. You may use fewer or more sections as appropriate, or even combine related information. However, remember: each section must be unique and non-repetitive.

10. REFERENCES AND CITATIONS:
${reference_info}
   - Check for a References or Bibliography section in the paper text
   - Report the number of citations/references found (e.g., "The paper cites X references" or "The paper includes a bibliography with approximately X entries")
   - If references are numbered [1], [2], etc., count the highest number you see
   - If no references section is found after checking the entire paper text, explicitly state "No references section found in the provided text"
   - Do NOT claim there are no references unless you have verified by searching the entire paper text for a References/Bibliography section
   - If the paper cites many works (50+), mention this as it indicates the breadth of related work covered
   - Include a brief "References" section at the end of your summary reporting the citation count

11. TECHNICAL COMPREHENSIVENESS - CRITICAL FOR DEEP ANALYSIS:
   - Include KEY EQUATIONS: Mathematical formulations central to the paper's contribution, with notation explained (e.g., "The free energy is defined as F = U - TS where U is internal energy, T is temperature, S is entropy")
   - Describe ALGORITHMS: Step-by-step algorithmic approaches, not just high-level descriptions (e.g., "The algorithm first computes X, then optimizes Y using method Z")
   - Document MATHEMATICAL FOUNDATIONS: Theoretical basis, assumptions, and mathematical framework underlying the approach
   - Detail EXPERIMENTAL SETUPS: Specific configurations, parameters, hardware, software, datasets used (e.g., "Experiments used dataset X with N samples, trained for Y epochs with learning rate Z")
   - Provide COMPARISONS: Quantitative comparisons with baseline methods (include specific metrics: "Method A achieved 94.2% accuracy vs 87.5% for baseline B")
   - Explain MECHANISMS: How the proposed method works at a technical level, including the mathematical or algorithmic steps, not just what it does
   - Include COMPLEXITY ANALYSIS: Computational complexity, time/space requirements if discussed (e.g., "O(n log n) time complexity", "requires O(n²) memory")
   - Document VALIDATION: How results were validated, statistical tests used, significance levels (e.g., "Results validated using t-test with p < 0.001")
   - Include ARCHITECTURE DETAILS: For ML/neural network papers, describe model architecture, layers, activation functions, training procedures
   - Be COMPREHENSIVE: Include technical details even if they seem dense - readers need this information for deep understanding
   - Avoid surface-level descriptions: Instead of "the method uses optimization", say "the method uses gradient descent with learning rate 0.001 and momentum 0.9, optimizing the loss function L = ..."

Begin your summary now. Remember: NO repeated sections, NO repeated paragraphs, extract SPECIFIC claims with CONCRETE details, include TECHNICAL DEPTH (equations, algorithms, experimental details), and accurately report the number of references. Aim for 600-1000 words of comprehensive technical content:"""
    
    def render(
        self,
        title: str,
        authors: str,
        year: str,
        source: str,
        text: str,
        domain: Optional[str] = None,
        domain_instructions: Optional[str] = None,
        reference_count: Optional[int] = None,
        references_section_found: bool = False
    ) -> str:
        """Render template with optional domain-specific instructions and reference info.
        
        Args:
            title: Paper title.
            authors: Author names.
            year: Publication year.
            source: Source database.
            text: Paper text content.
            domain: Detected domain (e.g., "physics", "computer_science").
            domain_instructions: Domain-specific instructions.
            reference_count: Number of references detected (if available).
            references_section_found: Whether references section was found.
            
        Returns:
            Rendered prompt string.
        """
        domain_info = ""
        if domain:
            domain_info = f"Detected Domain: {domain}\n"
        
        domain_instructions_text = domain_instructions or (
            "   - For PHYSICS papers: Highlight specific equations, experimental parameters, energy scales, detection methods, and statistical significance\n"
            "   - For COMPUTER SCIENCE papers: Detail algorithms, complexity analysis, dataset characteristics, performance metrics, and comparisons\n"
            "   - For BIOLOGY papers: Include species, sample sizes, statistical methods, biological mechanisms, and experimental conditions\n"
            "   - For MATHEMATICS papers: Cover theorems, proofs, mathematical objects, computational complexity, and theoretical implications"
        )
        
        # Build reference info text
        if reference_count is not None and reference_count > 0:
            reference_info = f"   - DETECTED: The paper contains approximately {reference_count} references (based on citation numbering [1], [2], etc.)\n"
            reference_info += f"   - You MUST report this in your summary: \"The paper cites {reference_count} references\" or similar\n"
        elif references_section_found:
            reference_info = "   - DETECTED: A References/Bibliography section was found in the paper, but exact count could not be determined\n"
            reference_info += "   - You should search the paper text for the References section and count the entries\n"
        else:
            reference_info = "   - No references section was automatically detected, but you should still check the entire paper text\n"
            reference_info += "   - Look for sections titled 'References', 'Bibliography', or 'Works Cited'\n"
            reference_info += "   - Only claim 'no references' if you have verified the entire paper text contains no such section\n"
        
        # Extract key terms from title for topic validation
        import re
        # Extract significant words from title (3+ characters, not common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        key_terms = [word for word in title_words if word not in stop_words][:10]  # Top 10 key terms
        key_terms_text = ", ".join(key_terms) if key_terms else "key terms from the title"
        
        # Replace template variables
        prompt = self.template_str.replace("${domain_info}", domain_info).replace(
            "${domain_instructions}", domain_instructions_text
        ).replace("${reference_info}", reference_info).replace("${title}", title).replace(
            "${authors}", authors
        ).replace("${year}", year).replace("${source}", source).replace("${text}", text)
        
        # Add key terms hint to topic validation section
        prompt = prompt.replace(
            "Extract 5-10 key terms from the paper title and abstract",
            f"Extract 5-10 key terms from the paper title and abstract (example key terms from title: {key_terms_text})"
        )
        
        return prompt


class LiteratureReviewSynthesis(ResearchTemplate):
    """Template for synthesizing multiple paper summaries into a literature review.

    Generates a literature review that identifies themes, compares approaches,
    and highlights key findings across multiple papers.
    """
    template_str = """=== PAPER SUMMARIES ===

${summaries}

=== END PAPER SUMMARIES ===

TASK: Write a literature review paragraph that synthesizes these ${num_papers} papers.

REQUIREMENTS:
1. Focus on the ${focus} aspects of these papers
2. Identify common themes, approaches, and findings
3. Compare and contrast different methods or results
4. Highlight gaps or areas needing further research
5. Write in academic prose (300-500 words)
6. Reference specific papers by their titles
7. Do NOT invent details not present in the summaries
8. Structure as a flowing narrative, not bullet points

Begin your literature review synthesis:"""


class ScienceCommunicationNarrative(ResearchTemplate):
    """Template for creating science communication narratives from research papers.

    Transforms technical research findings into understandable narratives
    for different audiences (general public, students, etc.).
    """
    template_str = """=== RESEARCH PAPERS ===

${papers}

=== END RESEARCH PAPERS ===

TASK: Create a science communication narrative that explains the key findings from these ${num_papers} research papers.

AUDIENCE: ${audience}
STYLE: ${narrative_style}

REQUIREMENTS:
1. Explain the scientific concepts clearly
2. Connect the research to real-world implications
3. Use storytelling techniques appropriate to the chosen style
4. Maintain scientific accuracy while being engaging
5. Include concrete examples from the papers
6. Write 600-800 words suitable for the target audience
7. Do NOT oversimplify to the point of inaccuracy
8. Make the science relatable and interesting

Begin your science communication narrative:"""


class ComparativeAnalysis(ResearchTemplate):
    """Template for comparison across multiple research papers.

    Provides comparison of methods, results, datasets, or other aspects
    across multiple papers in the same research area.
    """
    template_str = """=== PAPERS FOR COMPARISON ===

${papers}

=== END PAPERS ===

TASK: Perform a comparative analysis of these ${num_papers} papers focusing on their ${aspect}.

REQUIREMENTS:
1. Compare how the papers approach the ${aspect}
2. Identify similarities and differences
3. Evaluate strengths and weaknesses of each approach
4. Discuss implications of different approaches
5. Write 500-700 words in analytical academic style
6. Use specific examples from each paper
7. Structure with clear sections (Introduction, Comparison, Analysis, Conclusions)
8. Base all analysis on information in the provided papers

Begin your comparative analysis:"""


class ResearchGapIdentification(ResearchTemplate):
    """Template for identifying research gaps from literature analysis.

    Analyzes a set of papers to identify unanswered questions, methodological gaps,
    and areas needing further research.
    """
    template_str = """=== LITERATURE FOR GAP ANALYSIS ===

${papers}

=== END LITERATURE ===

TASK: Analyze these ${num_papers} papers to identify research gaps in the ${domain} domain.

REQUIREMENTS:
1. Identify unanswered questions or unexplored areas
2. Find methodological gaps or limitations
3. Spot inconsistencies or contradictory findings
4. Suggest directions for future research
5. Write 400-600 words with specific recommendations
6. Reference specific papers and their findings
7. Prioritize gaps that are important and feasible to address
8. Structure with sections (Current State, Identified Gaps, Recommendations)

Begin your research gap analysis:"""


class CitationNetworkAnalysis(ResearchTemplate):
    """Template for analyzing citation relationships and networks between papers.

    Examines how papers reference and build upon each other, identifying
    key works, research trajectories, and intellectual connections.
    """
    template_str = """=== PAPERS FOR NETWORK ANALYSIS ===

${papers}

=== END PAPERS ===

TASK: Analyze the citation network and intellectual connections between these ${num_papers} papers.

REQUIREMENTS:
1. Identify how papers build upon or contradict each other
2. Find common methodologies, datasets, or theoretical frameworks
3. Trace research trajectories or evolution of ideas
4. Identify key foundational papers or influential works
5. Write 500-700 words with clear analysis structure
6. Reference specific connections between papers
7. Discuss implications for the research field
8. Use network concepts (hubs, clusters, bridges) where appropriate

Begin your citation network analysis:"""


class ClaimsQuotesExtraction(ResearchTemplate):
    """Template for extracting key claims and important quotes from research papers.
    
    Identifies main claims, hypotheses, findings, and extracts important direct quotes
    with proper context and attribution.
    """
    template_str = """=== PAPER CONTENT ===

Title: ${title}
Authors: ${authors}
Year: ${year}
Source: ${source}

PAPER TEXT:
${text}

=== END PAPER CONTENT ===

TASK: Extract key claims, hypotheses, findings, and important direct quotes from this research paper.

REQUIREMENTS:

1. KEY CLAIMS AND HYPOTHESES:
   - Identify the main claims the paper makes (what does it claim to show or prove?)
   - Extract the primary hypotheses or research questions
   - List the key findings and conclusions
   - Include claims about methodology, results, or theoretical contributions
   - Format each claim clearly with context

2. IMPORTANT DIRECT QUOTES:
   - Extract 5-10 of the most important direct quotes from the paper
   - QUOTE EXTRACTION AND FORMATTING:
     * Extract quotes VERBATIM from the paper text - do NOT modify or "correct" them
     * Extract quotes exactly as they appear in the source text
     * Preserve all aspects of the quote exactly as written, including spacing
     * Use proper quotation marks: "quote text" (double quotes)
     * CRITICAL: Only extract quotes that actually appear in the paper text - do NOT generate or invent quotes
   - Include quotes that:
     * State key findings or results
     * Express main hypotheses or claims
     * Provide critical methodological insights
     * Summarize key contributions
     * Present important conclusions
   - For each quote, provide:
     * The exact quote text in quotation marks (extracted verbatim from source)
     * Context about where it appears (section name if available)
     * Brief explanation of why it's important
   - QUOTE FORMATTING STANDARD:
     * Use double quotation marks: "exact quote text"
     * Attribution format: "The authors state: \"quote text\"" OR "According to the paper: \"quote text\""
     * Vary attribution phrases to avoid repetition
     * Include section context when available: "In the Introduction, the authors state: \"quote text\""
     * Ensure proper spacing around quotes and punctuation
   - QUOTE FORMATTING ERRORS TO AVOID:
     * Do NOT put attribution inside quote marks:
       BAD: "According to the paper: 'actual quote'"
       GOOD: According to the paper: "actual quote"
     * Always include quote text (never just context)
     * Remove duplicate quotes before submitting

3. STRUCTURE YOUR OUTPUT AS:
   ## Key Claims and Hypotheses
   [List of main claims, one per bullet point or numbered item]
   
   ## Important Quotes
   [For each quote:]
   **Quote:** "[exact quote text]"
   **Context:** [section or location where quote appears]
   **Significance:** [why this quote is important]

4. ACCURACY REQUIREMENTS:
   - ONLY extract quotes that appear verbatim in the paper text
   - Do NOT paraphrase or modify quotes - extract them exactly as written
   - Do NOT invent claims not present in the paper
   - Include section references when possible (e.g., "Introduction", "Results", "Discussion")
   - If you cannot find a specific claim or quote, do not include it

5. DEDUPLICATION REQUIREMENTS:
   - Each quote appears only ONCE in your output
   - Each claim appears only ONCE in your output
   - Before adding a quote, check if you've already included it
   - If a quote appears in multiple sections, include it once with the most relevant context
   - Remove any duplicate quotes or claims before finalizing your output

6. COMPREHENSIVENESS:
   - Aim to extract 5-15 key claims
   - Include 5-10 important quotes
   - Cover all major aspects: methodology, results, conclusions, contributions
   - Prioritize claims and quotes that are central to the paper's contribution

7. FORMATTING:
   - Use clear markdown structure
   - Use quotation marks for all direct quotes
   - Use bullet points or numbered lists for claims
   - Maintain academic tone
   - Follow the QUOTE FORMATTING STANDARD above for consistent formatting

Begin extracting claims and quotes now:"""

    def render(
        self,
        title: str,
        authors: str,
        year: str,
        source: str,
        text: str
    ) -> str:
        """Render template for claims and quotes extraction.
        
        Args:
            title: Paper title.
            authors: Author names.
            year: Publication year.
            source: Source database.
            text: Paper text content.
            
        Returns:
            Rendered prompt string.
        """
        return self.template_str.replace("${title}", title).replace(
            "${authors}", authors
        ).replace("${year}", year).replace("${source}", source).replace("${text}", text)


class MethodsToolsAnalysis(ResearchTemplate):
    """Template for analyzing methods, algorithms, frameworks, datasets, and tools used in research papers.
    
    Extracts detailed information about:
    - Algorithms and methodologies
    - Software frameworks and libraries
    - Datasets employed
    - Evaluation metrics
    - Software tools and platforms
    """
    template_str = """=== PAPER CONTENT ===

Title: ${title}
Authors: ${authors}
Year: ${year}
Source: ${source}

PAPER TEXT:
${text}

=== END PAPER CONTENT ===

PROFESSIONAL TONE REQUIREMENTS:
- Begin directly with the analysis - NO conversational openings
- Do NOT use phrases like: "Okay, here's...", "Here's the analysis...", 
  "Let me analyze...", "I'll extract...", "Okay, let's begin...", or similar conversational language
- Start immediately with substantive content in formal academic tone
- Example BAD: "Okay, here's the analysis of methods and tools..."
- Example GOOD: "## Algorithms and Methodologies\n\n*   Free Energy Principle (exact quote from paper) – \"quote text\""

TASK: Analyze and extract detailed information about the methods, algorithms, frameworks, datasets, evaluation metrics, and software tools used in this research paper.

CRITICAL ANTI-BLOAT INSTRUCTIONS (ENFORCED):
- If a section has NO items found after searching the paper, write EXACTLY ONE LINE: "Not specified in paper"
- NEVER list individual placeholder items with "Not specified in paper" - this creates bloat
- MAXIMUM 3 sections total can be "Not specified in paper" across the entire output
- If you find yourself writing "Not specified" more than 3 times, STOP and revise your search
- Keep total output UNDER 200 lines and UNDER 1500 words - be concise
- Do NOT repeat the same information in multiple sections

VERIFICATION CHECKLIST (Execute before responding):
1. Search the FULL paper text for each tool/framework name you want to list
2. Only include if EXACT NAME appears in the paper text
3. Count "Not specified" instances (must be ≤3 total)
4. Count total output lines (must be ≤200)
5. Count total words (must be ≤1500)
6. Remove any duplicate items across sections

REQUIREMENTS:

1. ALGORITHMS AND METHODOLOGIES:
   - Identify all algorithms mentioned (e.g., gradient descent, active inference, PCA, etc.)
   - Document the specific methodology or approach used
   - Include algorithmic details: steps, parameters, configurations
   - Note any novel algorithms or modifications to existing ones
   - Include mathematical formulations if central to the method

2. SOFTWARE FRAMEWORKS AND LIBRARIES:
   - Extract all software frameworks mentioned (e.g., PyTorch, TensorFlow, JAX, scikit-learn, etc.)
   - List programming languages used (Python, MATLAB, R, etc.)
   - Identify libraries and packages (NumPy, Pandas, etc.)
   - Note any custom implementations or code repositories mentioned
   - Include version numbers if specified

3. DATASETS:
   - List all datasets used (e.g., ImageNet, CIFAR-10, UCI datasets, etc.)
   - Include dataset characteristics: size, number of samples, features
   - Note any custom or proprietary datasets
   - Document data preprocessing steps if mentioned
   - Include data sources or repositories

4. EVALUATION METRICS:
   - Extract all metrics used for evaluation (accuracy, F1-score, precision, recall, MSE, etc.)
   - Include statistical tests (t-tests, ANOVA, etc.) and significance levels
   - Note baseline comparisons and benchmarks
   - Document performance measures and their values
   - Include any custom metrics defined in the paper

5. SOFTWARE TOOLS AND PLATFORMS:
   - Identify software tools used (MATLAB, SPSS, R, etc.)
   - List platforms or environments (Google Colab, AWS, local clusters, etc.)
   - Note hardware specifications if mentioned (GPUs, CPUs, memory)
   - Include any simulation tools or software packages
   - Document computational resources used

6. STRUCTURE YOUR OUTPUT AS:
   ## Algorithms and Methodologies
   [List of algorithms with descriptions]
   
   ## Software Frameworks and Libraries
   [List of frameworks, languages, and libraries]
   
   ## Datasets
   [List of datasets with characteristics]
   
   ## Evaluation Metrics
   [List of metrics and evaluation methods]
   
   ## Software Tools and Platforms
   [List of tools, platforms, and computational resources]

7. CRITICAL ACCURACY REQUIREMENTS - NO SPECULATION ALLOWED (ENHANCED):
   - ONLY list items EXPLICITLY NAMED in the paper text
   - DO NOT list items that are NOT found in the paper - this is CRITICAL
   - DO NOT create entries for common items (Accuracy, F1-score, Precision, Recall, MSE, Google Colab, AWS, MATLAB, R, etc.) if they are not explicitly mentioned
   - For each item you list, provide the EXACT QUOTE or specific text from the paper that mentions it
   - Format: "Item Name (exact quote from paper) – \"quote text\"" or "Item Name (exact quote from paper) – mentioned in [section]"
   - NEVER infer tools based on methodology (e.g., don't assume PyTorch for deep learning)
   - Verification: Before listing any item, search the paper text for its explicit mention
   - NEVER use words like: "implied", "likely", "potentially", "not explicitly mentioned but", "probably", "may have been used", "suggested", "inferred", "assumed"
   - NEVER guess or infer tools, frameworks, datasets, or methods based on context or prior knowledge
   - If a tool/framework/dataset is not EXPLICITLY NAMED in the paper, DO NOT include it in your list
   - Example format: "PyTorch (exact quote from paper) – \"PyTorch version 1.8.0\"" - ONLY if the paper explicitly states "PyTorch version 1.8.0"
   - Include specific names, versions, and details ONLY when explicitly stated in the paper
   - Be precise with technical terminology - use exact terms from the paper

8. COMPREHENSIVENESS AND SECTION-LEVEL REPORTING:
   - Extract ALL relevant tools, frameworks, datasets, and methods that are EXPLICITLY MENTIONED
   - Include both standard and custom solutions that are EXPLICITLY NAMED
   - Cover all aspects: algorithms, software, data, evaluation, hardware - but ONLY if explicitly stated
   - Prioritize items that are central to reproducing the research
   - CRITICAL: If a section has no items found, write "Not specified in paper" at the SECTION LEVEL, not for individual items
   - DO NOT list individual items with "Not specified in paper" - this is incorrect
   - CORRECT format for empty section: "## Evaluation Metrics\n\nNot specified in paper"
   - INCORRECT format: "*   Accuracy (exact quote from paper) – \"Not specified in paper\""
   - If you find items, list them with quotes. If you find no items, state "Not specified in paper" for the entire section

9. FORMATTING:
   - Use clear markdown structure with sections
   - Use bullet points or numbered lists within sections
   - Include specific details (versions, sizes, names) ONLY when explicitly stated in the paper
   - Maintain technical accuracy - quote exact text from the paper when possible
   - Format: For each item, include the exact quote or specific text that mentions it
   - Format: "Item Name (exact quote from paper)" or "Item Name - mentioned in [section]"

10. VERIFICATION (CRITICAL):
    - Before listing any item, verify it appears explicitly in the paper text
    - Search the paper text for the explicit mention of each item before including it
    - If you're unsure whether something is explicitly stated, DO NOT include it
    - When in doubt, exclude rather than speculate
    - Every item must have a clear, explicit mention in the paper text
    - Do NOT list items that are "commonly used" or "typically associated" - only list what is explicitly named

11. FORMATTING EXAMPLES - CORRECT vs INCORRECT:

CORRECT FORMAT (items found):
## Evaluation Metrics

*   t-tests (exact quote from paper) – "statistical tests (t-tests, ANOVA, etc.) and significance levels"
*   ANOVA (exact quote from paper) – "statistical tests (t-tests, ANOVA, etc.) and significance levels"

## Software Frameworks and Libraries

*   PyTorch (exact quote from paper) – "PyTorch version 1.8.0"

INCORRECT FORMAT (DO NOT DO THIS):
## Evaluation Metrics

*   Accuracy (exact quote from paper) – "Not specified in paper"
*   F1-score (exact quote from paper) – "Not specified in paper"
*   Precision (exact quote from paper) – "Not specified in paper"
*   Recall (exact quote from paper) – "Not specified in paper"
*   Mean Squared Error (MSE) (exact quote from paper) – "Not specified in paper"
*   t-tests (exact quote from paper) – "statistical tests (t-tests, ANOVA, etc.) and significance levels"

CORRECT FORMAT (no items found):
## Evaluation Metrics

Not specified in paper

## Software Tools and Platforms

Not specified in paper

Begin analyzing methods and tools now. Remember: ONLY list items that ARE found, DO NOT list items that are NOT found, and use section-level "Not specified in paper" when no items are found:"""

    def render(
        self,
        title: str,
        authors: str,
        year: str,
        source: str,
        text: str
    ) -> str:
        """Render template for methods and tools analysis.
        
        Args:
            title: Paper title.
            authors: Author names.
            year: Publication year.
            source: Source database.
            text: Paper text content.
            
        Returns:
            Rendered prompt string.
        """
        return self.template_str.replace("${title}", title).replace(
            "${authors}", authors
        ).replace("${year}", year).replace("${source}", source).replace("${text}", text)


class PaperClassificationTemplate(ResearchTemplate):
    """Template for classifying papers by research type."""
    
    template_str = """=== PAPER CONTENT ===

Title: ${title}
Authors: ${authors}
Year: ${year}

FULL PAPER TEXT:
${full_text}

=== END PAPER CONTENT ===

TASK: Classify this research paper based on its primary contribution and focus.

CLASSIFICATION CATEGORIES:

1. **Core/Theory/Math**: Papers focused on theoretical foundations, mathematical proofs, fundamental principles, or core algorithmic/theoretical contributions. These papers advance understanding of fundamental concepts rather than applying them.

2. **Translation/Tool Development**: Papers that develop software tools, frameworks, libraries, or systems that translate research into usable implementations. These papers focus on tooling, infrastructure, or software development.

3. **Applied**: Papers that apply existing theories, methods, or tools to solve real-world problems in a specific domain. These papers demonstrate practical applications.

REQUIREMENTS:
- Consider the ENTIRE paper content, not just the abstract
- Identify the PRIMARY contribution and focus
- If Applied, specify the domain (e.g., "Computer Science", "Biology", "Medicine", "Physics", "Engineering", etc.)
- Provide confidence score (0.0 to 1.0)
- Include brief reasoning

OUTPUT FORMAT (JSON):
{
  "category": "core_theory_math" | "translation_tool" | "applied",
  "domain": "Domain name" (required if category is "applied", null otherwise),
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}
"""
    
    def render(
        self,
        title: str,
        authors: str,
        year: str,
        full_text: str
    ) -> str:
        """Render template for paper classification.
        
        Args:
            title: Paper title.
            authors: Author names.
            year: Publication year.
            full_text: Full paper text content.
            
        Returns:
            Rendered prompt string.
        """
        return self.template_str.replace("${title}", title).replace(
            "${authors}", authors
        ).replace("${year}", year).replace("${full_text}", full_text)

