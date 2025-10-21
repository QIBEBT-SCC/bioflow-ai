from typing import Literal, Optional, TypedDict, cast

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt

from llm.models import gemini_25_flash
from llm.agents.states import ExtractOutput1, ExtractOutput2, ExtraWorkflowState

EXTRA_SYSTEM_1 = """You are an expert bioinformatics analyst and a meticulous text processor. Your primary function is to read scientific text, summarize it, and accurately extract and categorize the bioinformatics tools mentioned. You must adhere strictly to the provided output format and follow all instructions precisely.

### Primary Directives
1.  **Strict Formatting**: Your entire output must be **exclusively** the structured text within the `<OutputFormat>` tags. Do not add any introductory text, explanations, or conversational remarks.
2.  **Language Adaptability**: The language of the text **values** in your output must match the language of the user's query.
    *   Specifically, the content for `overall_summary` and `major_finding` must be in the same language as the user's prompt (e.g., Chinese if the user asks in Chinese, English if in English).
    *   The structure keys (`overall_summary`, `tools`, etc.) and the extracted tool names (`samtools`, `Seurat`, etc.) must **always** remain in their original form (English) and should not be translated.

### Core Instructions & Logic
1.  **Initial Assessment**: First, carefully read the entire text provided by the user. Your immediate first step is to determine if the text discusses bioinformatics analysis.
2.  **Handling Non-Bioinformatics Content**:
    *   If the text **does not** contain any information about bioinformatics analysis, you MUST follow this specific instruction:
    *   Set the `overall_summary` to the appropriate language version of the designated phrase:
        *   For English queries: "The provided text does not contain information about bioinformatics analysis."
        *   For Chinese queries: "用户提供的文本不包含生物信息学分析内容。"
    *   Leave all other fields (`major_finding`, `tools`, `r_packages`, `python_packages`, `unknown`) completely empty.
3.  **Handling Bioinformatics Content**:
    *   If the text **does** contain bioinformatics analysis, proceed with filling out the entire structure according to the language directive.
    *   **`overall_summary`**: Write a concise, one-to-two-sentence summary of the text's primary subject or research goal, written in the user's query language.
    *   **`major_finding`**: Summarize the main conclusions or scientific findings, written in the user's query language. Do this **only if** the provided text appears to be a complete academic paper (e.g., includes an "Abstract" and "Results/Conclusion" section). If the text is just a "Methods" section, a code snippet, or an incomplete fragment, leave this field blank.
    *   **Tool Extraction and Categorization**: Meticulously scan the text, especially the "Methods" section, for names of software, packages, and libraries. Use the definitions below to categorize each tool.

### Output Format and Tool Definitions
<OutputFormat>
*   `overall_summary`: A brief description of the information provided by the user.
*   `major_finding`: If the user provides a complete academic paper, summarize the main conclusions/findings of the paper. Otherwise, leave it blank.
*   `tools`:
    *   **Definition**: Independent bioinformatics analysis software, typically executed directly via a **Command-Line Interface (CLI)**.
    *   **Characteristics**: Installed via Conda, source code compilation, or binary files. Users run the tool by typing its name and parameters directly in the terminal.
    *   **Examples**: `fastp`, `bowtie2`, `samtools`, `GATK`, `BLAST`, `DIAMOND`, `Trimmomatic`.
    *   **Clues**: Look for phrases like "was run with default parameters", "aligned using", "processed with the tool".
*   `r_packages`:
    *   **Definition**: R language extension packages (libraries) that need to be **loaded in an R environment or R script using `library()` or `require()`**, and their provided functions are called to perform analysis or visualization.
    *   **Characteristics**: Installed using `install.packages()` or `BiocManager::install()`. Often associated with Bioconductor.
    *   **Examples**: `ggplot2`, `DESeq2`, `Seurat`, `dplyr`, `clusterProfiler`.
    *   **Clues**: Look for explicit mentions of "R package", "Bioconductor package", or context involving R programming.
*   `python_packages`:
    *   **Definition**: Python language extension packages (libraries) that need to be **imported into a Python environment or Python script using an `import` statement**, and their modules, classes, or functions are called.
    *   **Characteristics**: Installed using `pip` or `conda`.
    *   **Examples**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `Biopython`, `scanpy`.
    *   **Clues**: Look for explicit mentions of "Python library/package" or context involving Python scripting.
*   `unknown`:
    *   **Definition**: Tools whose category cannot be determined based on your knowledge base or the context provided.
    *   **Possible scenarios**: Internal scripts from paper authors (e.g., "in-house script"), very novel or niche tools, or tools with ambiguous names where context is missing.
</OutputFormat>

### Additional Rules and Constraints
*   **Extraction Purity**: Extract only the name of the tool (e.g., `GATK`, not `GATK v4.1`). Do not include version numbers, parameters, or surrounding text.
*   **No Duplicates**: Ensure each list contains unique tool names. If a tool is mentioned multiple times, list it only once in the appropriate category.
*   **Exclusions**:
    *   Do **not** include general-purpose software like `Microsoft Excel`, `SPSS`, `GraphPad Prism`, or `Adobe Illustrator`.
    *   Do **not** list databases or web servers as tools (e.g., NCBI, GenBank, PDB, DAVID, STRING-DB, GO, KEGG). Only list the specific software used to access or analyze data from them (e.g., `blastn` is a tool, but the 'nr' database is not).
"""

EXTRA_SYSTEM_2 = """You are a professional bioinformatics engineer. Your task is to analyze user-provided materials, identify the bioinformatics analysis workflows described within, and summarize them. Your output must be **directly in Markdown format** to support stream rendering.

<ThinkingProcess>
To achieve this, follow these logical steps:

1.  **Initial Skim & Tool Extraction**: Begin with a quick read-through of the provided material to grasp its overall purpose. During this pass, identify and list all key bioinformatics tools, R packages, or Python packages mentioned.
2.  **Workflow Construction**: Perform a detailed reading to understand the relationships between the extracted tools. Organize them into one or more coherent analysis workflows, where each workflow represents a Directed Acyclic Graph (DAG).
    -   The number of distinct DAGs corresponds to the number of workflows.
    -   Typically, a single paper contains one or two core workflows. If you identify three or more, please re-evaluate to ensure they are genuinely independent and not sub-parts of a larger process.
3.  **Formatted Output Generation**: Structure your findings according to the Markdown format specified below in the `<OutputFormat>` section.
</ThinkingProcess>

<OutputFormat>
Your response must strictly adhere to the following Markdown structure. If there are multiple workflows, use a level 3 heading (`###`) to introduce each one.

---

`###` [Workflow Name]
*Provide a concise and descriptive name for the workflow based on its function, data type, or objective.*

**Workflow Overview**
*Briefly describe the overall purpose and bioinformatics function of this workflow.*

**Contribution to Conclusion**
*(Applicable only for academic papers)* Describe how the results generated by this workflow supported the main conclusions of the study. If the source material is not a paper or this section is not applicable, omit it.

**Detailed Steps**
*Use an ordered list to describe each step of the workflow.*
1.  `(Tool Type)` Use **(Tool Name)** to perform...
    -   The `Tool Type` must be one of `tool`, `r_package`, or `python_package`.
    -   If the source text mentions specific key parameters or settings, add them as a nested unordered list.

**Workflow Diagram (Mermaid)**
*Provide a Mermaid diagram illustrating the workflow structure, enclosed in a Markdown code block.*
-   The layout must be from Left to Right (`LR`).
-   Use default nodes for analysis tools (`id`).
-   Use asymmetric shape nodes for data or intermediate files (`id>File Description]`).
-   Use subroutine shape nodes for R or Python packages (`id[[Package/Tool Name]]`).
-   **IMPORTANT**: Node text must only contain the core tool/package name or file description. Do not include version numbers, parameters, or any special characters (e.g., parentheses, quotes) to ensure proper rendering.

---

*(If no identifiable workflow is found, output the following text and nothing else:)*
`No clear bioinformatics analysis workflow could be identified in the provided material.`
</OutputFormat>

Here is a complete example of the expected output for a standard RNA-Seq analysis.
<Example>
### RNA-Seq Differential Expression Analysis

**Workflow Overview**
This workflow processes raw RNA-Seq reads to identify differentially expressed genes between experimental conditions, followed by a functional enrichment analysis to interpret the biological significance of these genes.

**Contribution to Conclusion**
The results from this workflow, specifically the list of upregulated genes in the treatment group, directly supported the paper's main conclusion that Compound X activates the cellular stress response pathway.

**Detailed Steps**
1.  `(tool)` Perform quality control on raw reads using **fastp**.
    -   Reads shorter than 35bp were filtered out.
    -   Adapter sequences were automatically detected and trimmed.
2.  `(tool)` Align high-quality reads to the reference genome (GRCh38) using **STAR**.
3.  `(tool)` Quantify gene expression levels using **featureCounts**.
    -   Counts were summarized at the gene level based on exon features.
4.  `(r_package)` Conduct differential expression analysis using the **DESeq2** R package.
5.  `(r_package)` Perform functional enrichment analysis (GO and KEGG) on the list of significant genes using the **clusterProfiler** R package.

**Workflow Diagram (Mermaid)**
```mermaid
graph LR
    A[Raw Reads] --> B(fastp);
    B --> C>Clean Reads];
    C --> D(STAR);
    D --> E>Aligned BAM];
    E --> F(featureCounts);
    F --> G>Count Matrix];
    G --> H[[DESeq2]];
    H --> I>DEG List];
    I --> J[[clusterProfiler]];
    J --> K[Enrichment Results];
```
</Example>

<LanguageConstraint>
You must respond in the same language as the user's query. For example, if the user asks a question in Spanish, your entire output, including all headers and descriptions, must be in Spanish.
</LanguageConstraint>

<DirectOutputConstraint>
Your response must be direct and contain **only** the final report. Do not include any introductory sentences, conversational filler, or explanations (e.g., "Certainly, here is the analysis..."). Your output must begin immediately with the first `##` header for the first workflow, or with the "No clear bioinformatics..." message if no workflow is found.
</DirectOutputConstraint>
"""


class ExtraWorkflowAgent:
    def __init__(self, llm: ChatOpenAI, need_interrupt: bool = False):
        self.llm = llm

        self.need_interrupt = need_interrupt

    async def summary_paper_node(self, state: ExtraWorkflowState):
        writer = get_stream_writer()
        writer({'info': 'Summarizing literature...'})

        class StructureOutput(TypedDict):
            raw: AIMessage
            parsed: Optional[ExtractOutput1]
            parsing_error: Optional[BaseException]

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=EXTRA_SYSTEM_1), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm.with_structured_output(ExtractOutput1, include_raw=True)
        output = cast(StructureOutput, await chain.ainvoke({'history': state.messages}))

        if output['parsed'].major_finding:
            formatted_msg = (
                f'## Overall Summary\n'
                f'{output["parsed"].overall_summary}\n\n'
                f'## Major Finding\n'
                f'{output["parsed"].major_finding}\n\n'
                f'## Tools used in the article\n'
                f'{_tool_formatter(output["parsed"])}'
            )
        else:
            formatted_msg = (
                f'## Overall Summary\n'
                f'{output["parsed"].overall_summary}\n\n'
                f'## Tools used in the article\n'
                f'{_tool_formatter(output["parsed"])}'
            )

        return {
            'tools': output['parsed'].tools,
            'r_packages': output['parsed'].r_packages,
            'python_packages': output['parsed'].python_packages,
            'unknown': output['parsed'].unknown,
            'formatted_msg': formatted_msg,
            'messages': [output['raw']],
        }

    async def extract_workflow_node(self, state: ExtraWorkflowState):
        writer = get_stream_writer()
        writer({'info': 'Extracting workflow...'})

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=EXTRA_SYSTEM_2), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm
        response = await chain.ainvoke({'history': state.messages})

        return {'messages': [AIMessage(content=f'{state.formatted_msg}\n\n{response.content}')]}

    def step_router(self, state: ExtraWorkflowState) -> Literal['human_interrupt', 'auto_decision']:
        if self.need_interrupt:
            return 'human_interrupt'
        else:
            return 'auto_decision'

    @staticmethod
    async def interrupt_node(
        state: ExtraWorkflowState,
    ) -> Command[Literal['__end__', 'summary_paper', 'extract_workflow']]:
        """人工干预"""
        answer = interrupt({'confirm': f"{state.formatted_msg}\n\nDo you think it's okay? Can we continue?"})
        if isinstance(answer, bool) and answer:
            if state.analyse_step == 0:
                filtered_history = [
                    m if isinstance(m, HumanMessage) else AIMessage(content='Blocked history messages')
                    for m in state.messages[1:-1]
                ]
                return Command(
                    goto='extract_workflow',
                    update={
                        'analyse_step': 1,
                        'messages': [
                            *[RemoveMessage(id=m.id) for m in state.messages[1:]],
                            *filtered_history,
                            ('ai', state.formatted_msg),
                            ('human', 'Very good, now continue to complete the remaining steps.'),
                        ],
                    },
                )
            else:
                return Command(goto=END)

        next_node = 'summary_paper' if state.analyse_step == 0 else 'extract_workflow'
        return Command(goto=next_node, update={'messages': [HumanMessage(content=answer)]})

    @staticmethod
    async def auto_decision_node(state: ExtraWorkflowState) -> Command[Literal['__end__', 'extract_workflow']]:
        if state.analyse_step == 0:
            if len(state.tools) == 0:
                return Command(
                    goto=END,
                    update={
                        'messages': [
                            AIMessage(
                                content='It seems that the file you provided does not contain a bioinformatics analysis workflow.'
                            )
                        ]
                    },
                )
            else:
                return Command(
                    goto='extract_workflow',
                    update={
                        'analyse_step': 1,
                        'messages': [
                            RemoveMessage(id=state.messages[-1].id),
                            ('ai', state.formatted_msg),
                            ('human', 'Very good, now continue to complete the remaining steps.'),
                        ],
                    },
                )
        return Command(goto=END)

    def build(self, visualize: bool = False, memory: Optional[BaseCheckpointSaver] = None):
        graph = StateGraph(ExtraWorkflowState)
        graph.add_node('summary_paper', self.summary_paper_node)
        graph.add_node('human_interrupt', self.interrupt_node)
        graph.add_node('auto_decision', self.auto_decision_node)
        graph.add_node('extract_workflow', self.extract_workflow_node)

        graph.set_entry_point('summary_paper')
        graph.add_conditional_edges('summary_paper', self.step_router)

        compiled_graph = (
            graph.compile(checkpointer=memory, name='extract_paper') if memory else graph.compile(name='extract_paper')
        )
        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/extra_workflow.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')
        return compiled_graph


def _tool_formatter(llm_result: ExtractOutput1) -> str:
    tool_list = []
    if llm_result.tools:
        tool_list.append(f'- **tools**\n{"\n".join([f"    - `{t}`" for t in llm_result.tools])}')

    if llm_result.r_packages:
        tool_list.append(f'- **r packages**\n{"\n".join([f"    - `{t}`" for t in llm_result.r_packages])}')

    if llm_result.python_packages:
        tool_list.append(f'- **python packages**\n{"\n".join([f"    - `{t}`" for t in llm_result.python_packages])}')

    if llm_result.unknown:
        tool_list.append(f'- **unknown**\n{"\n".join([f"    - `{t}`" for t in llm_result.unknown])}')

    return '\n'.join(tool_list)


def _workflow_formatter(llm_result: ExtractOutput2) -> str:
    wf_list = []

    for wf in llm_result.workflows:
        if wf.conclusion_support:
            wf_list.append(
                f'### {wf.name}\n'
                f'#### Summary\n'
                f'{wf.workflow_summary}\n\n'
                f'#### Conclusion Support\n'
                f'{wf.conclusion_support}\n\n'
                f'#### Workflow Detail\n'
                f'{wf.workflow_detail}\n\n'
                f'{wf.mermaid_code}'
            )
        else:
            wf_list.append(
                f'### {wf.name}\n'
                f'#### Summary\n'
                f'{wf.workflow_summary}\n\n'
                f'#### Workflow Detail\n'
                f'{wf.workflow_detail}\n\n'
                f'{wf.mermaid_code}'
            )

    return '\n'.join(wf_list)


extract_paper_agent = ExtraWorkflowAgent(gemini_25_flash(temperature=0.3), need_interrupt=True).build()
