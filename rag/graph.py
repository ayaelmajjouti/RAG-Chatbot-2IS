from typing import List, Dict, TypedDict, Tuple
import logging
import re 
import json
from base_rag import BaseRAG


from .openrouter_rag import OpenRouterRAG
from langgraph.graph import END, StateGraph
from vector_db.faiss_store import VectorStore
from scraper.content_processor import ContentProcessor

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """
    Represents the state of our graph, holding all data passed between nodes.

    Attributes:
        question: The user's current question.
        context: The formatted context string for the LLM.
        sources: The URLs of the retrieved documents.
        documents: The raw documents retrieved from the vector store.
        answer: The LLM-generated answer.
        history: The list of past user/assistant messages.
        classification : The classification of the user's input.
        is_curriculum_question: A boolean flag to track if the query is about the syllabus.
        is_list_request: A boolean flag to track if the query is a list request.
        evaluation_results: A dictionary to store the results from the evaluation node.

    """
    question: str
    context: str
    sources: List[str]
    documents: List[Dict]
    answer: str
    history: List[Dict]
    classification: str
    is_curriculum_question: bool
    is_list_request: bool
    evaluation_results: Dict


class RAGGraph:
    def __init__(self, vector_store: VectorStore, processor: ContentProcessor, rag_model: BaseRAG):
        """Initializes the RAG graph with necessary components."""
        self.vector_store = vector_store
        self.processor = processor
        self.rag_model = rag_model
        # Build and compile the graph when the class is instantiated
        self.workflow = self.build_graph()

    def _parse_evaluation(self, response: str) -> Tuple[int, str]:
        """
        A helper function to robustly parse the LLM's evaluation response,
        extracting the numerical score and the justification text.
        """
        score_match = re.search(r"Score:\s*(\d)", response, re.IGNORECASE)
        justification_match = re.search(r"Justification:\s*(.*)", response, re.DOTALL | re.IGNORECASE)

        score = int(score_match.group(1)) if score_match else 0
        justification = justification_match.group(1).strip() if justification_match else "No justification provided."
        return score, justification
    ### NODE FUNCTIONS ###
    def analyze_query(self, state: GraphState) -> GraphState:
        """
        A powerful multi-task node that performs classification, rewriting, and routing analysis in a single LLM call.
        This significantly reduces latency by combining multiple steps.
        """
        logger.info("---NODE: ANALYZE QUERY (MULTI-TASK)---")
        question = state["question"]
        history = state.get("history", [])
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        analysis_prompt = f"""You are an expert query analyzer. Your task is to analyze the user's "Follow-up Question" based on the "Chat History" and return a JSON object with three fields: "classification", "standalone_question", and "is_list_request".

1.  **classification**: Classify the question into one of three categories:
    - `on_topic_question`: For specific questions about the 2IS Master's program.
    - `off_topic_question`: For questions unrelated to the program.
    - `conversational`: For greetings, thanks, or meta-questions.

2.  **standalone_question**: Rewrite the "Follow-up Question" to be a complete, standalone question. If it's already standalone, return it as is.

3.  **is_list_request**: A boolean (true/false). Set to `true` if the user is asking to list multiple courses (e.g., "what are the courses in year 1?", "list teachers for M1S2"). Otherwise, set to `false`.

Chat History:
{history_str}

Follow-up Question: "{question}"

Respond with ONLY a valid JSON object in the following format and nothing else:
{{
  "classification": "...",
  "standalone_question": "...",
  "is_list_request": ...
}}
"""
        try:
            response_str = self.rag_model.query(prompt=analysis_prompt)
            # Find the JSON part of the response, robustly
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in response", response_str, 0)
            analysis_json = json.loads(json_match.group(0))

            logger.info(f"Query analysis complete: {analysis_json}")
            return {**state, **analysis_json}
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse analysis from LLM: {e}. Defaulting to standard RAG path.")
            # Fallback to a safe default if parsing fails
            return {**state, "classification": "on_topic_question", "standalone_question": question, "is_list_request": False}

    
    def conversational_response(self, state: GraphState) -> GraphState:
        """
        Node: Generate a conversational response for non-knowledge questions.
        """
        logger.info("---NODE: CONVERSATIONAL RESPONSE---")
        question = state["question"]
        history = state.get("history", [])

        # Use the LLM to generate a natural, empathetic, and guiding conversational response
        prompt = f"""You are a helpful and friendly assistant for the 2IS Master's program at Toulouse Capitole University. The user has said something conversational (e.g., a greeting, a thank you, or an emotional statement). Your goal is to respond with empathy and gently guide the conversation toward your purpose.

- If they say hello or ask if you can help, respond warmly and state your purpose.
- If they thank you, respond graciously.
- If they express frustration or sadness (e.g., "I don't feel good"), be empathetic. You can suggest that if their stress is related to choosing a master's program, you are there to provide information to help them.

User's message: "{question}"

Your empathetic and helpful response:"""

        answer = self.rag_model.query(prompt=prompt, history=history)
        
        return {**state, "answer": answer.strip(), "sources": []}

    def off_topic_response(self, state: GraphState) -> GraphState:
        """
        Node: Generate a polite response for off-topic questions.
        """
        logger.info("---NODE: OFF-TOPIC RESPONSE---")
        question = state["question"]
        history = state.get("history", [])


        prompt = f"""You are a helpful and friendly assistant for the 2IS Master's program at Toulouse Capitole University. The user has asked a question that is outside your scope of knowledge. Your task is to politely inform them that you can only answer questions about the 2IS program and invite them to ask about that. Be friendly and helpful, not robotic.

User's off-topic question: "{question}" """
        answer = self.rag_model.query(prompt=prompt, history=history)
        return {**state, "answer": answer.strip(), "sources": []}
    

    
        
    def find_and_list_courses(self, state: GraphState) -> GraphState:
        """
        A specialized "super-node" that handles requests to list all courses for a given period.
        It retrieves all syllabus data, filters it in Python, and then generates a tailored response.
        """
        logger.info("---NODE: FIND AND LIST COURSES (Specialist Path)---")
        user_question = state.get("standalone_question") or state["question"]
        
        # Step 1: Use LLM to extract structured parameters (period, details)
        extraction_prompt = f"""You are an expert query parser. Your task is to analyze the user's question and extract two pieces of information: the 'period' and the 'details' they are asking for.

1.  **periods**: Normalize the academic period(s) into a list of one or more of these formats: "year 1, semester 1", "year 1, semester 2", "year 1", "year 2", "all". If the user asks for S1 and S2, you can use ["year 1"].

2.  **details**: Identify the specific fields the user wants from the available list. The available fields are: `course_title`, `teachers`, `period`, `ects`, `contact_hours`, `prerequisites`, `course_description`. Always include `course_title`.

Examples:
- User question: "list the courses in the first year" -> {{"periods": ["year 1"], "details": ["course_title"]}}
- User question: "who are the teachers for M1S2?" -> {{"periods": ["year 1, semester 2"], "details": ["course_title", "teachers"]}}
- User question: "give me the ECTS for all courses" -> {{"periods": ["all"], "details": ["course_title", "ects"]}}

User Question: "{user_question}"

Respond with ONLY a valid JSON object in the following format:
{{
   "periods": ["..."],
   "details": ["..."]
}}"""
        
        try:
            response_str = self.rag_model.query(prompt=extraction_prompt)
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in extraction response", response_str, 0)
            params = json.loads(json_match.group(0))
            target_periods = params.get("periods", ["all"])
            desired_details = params.get("details", ["course_title"])
            logger.info(f"Extracted periods: '{target_periods}', details: {desired_details}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse extraction from LLM: {e}. Providing a fallback response.")
            answer = "I had trouble understanding the specifics of your request for a course list. Could you please try rephrasing it?"
            return {**state, "answer": answer, "sources": []}

        # Step 2: Retrieve ALL syllabus data directly from the source JSON for 100% accuracy.
        try:
            with open('Syllabus.json', 'r', encoding='utf-8') as f:
                all_syllabus_courses = json.load(f)
            logger.info(f"Successfully loaded {len(all_syllabus_courses)} courses from Syllabus.json")
        except FileNotFoundError:
            logger.error("Syllabus.json not found. Cannot perform specialist list search.")
            return {**state, "answer": "I can't access the detailed course syllabus right now to create a list.", "sources": ["Syllabus.json"]}

        # Step 3: Filter the courses algorithmically in Python for guaranteed completeness.
        matching_courses = []
        for course in all_syllabus_courses:
            course_period = course.get("period", "").lower().strip()
            # Check if the course period matches any of the target periods
            for target_period in target_periods:
                if target_period == "all" or target_period.lower().strip() in course_period:
                    matching_courses.append(course)
                    break # Avoid adding the same course multiple times
        
        logger.info(f"Found {len(matching_courses)} courses matching periods '{target_periods}'.")

        if not matching_courses:
            answer = f"I couldn't find any courses for the period(s) '{', '.join(target_periods)}'. Please check the year or semester and try again."
            return {**state, "answer": answer, "sources": ["Syllabus.json"]}

        # Step 4: Format the context for the final presentation by the LLM.
        # We extract only the desired details to keep the context clean and focused.
        context_for_llm_list = []
        for course in sorted(matching_courses, key=lambda x: x.get('course_title', '')):
            course_details = {key: course.get(key, "Not specified") for key in desired_details}
            context_for_llm_list.append(course_details)

        if context_for_llm_list:
            context = json.dumps(context_for_llm_list, indent=2)
            sources = ["local_json://Syllabus.json"]
        else:
            # Fallback logic
            logger.warning(f"No syllabus entries found for '{target_periods}'. Falling back to general document search.")
            flyer_docs = self.vector_store.similarity_search(self.processor.encode_query(user_question), k=5)
            context, sources = self.rag_model.format_context_with_sources(flyer_docs)
            if not context:
                answer = f"I'm sorry, I couldn't find any courses listed for '{', '.join(target_periods)}' in any of the available program documents."
                return {**state, "answer": answer, "sources": []}

        # Step 5: Generate the final answer using the correct RAG pattern.
        # We pass the user's question and the structured context separately.
        # The RAG model's query method will construct the final prompt.
        
        history = state.get("history", [])
        final_answer = self.rag_model.query(
            prompt=user_question,
            context=context,
            history=history,
            is_curriculum_question=True
        )
        
        return {**state, "answer": final_answer, "sources": sources, "context": context}        
    
    def retrieve(self, state: GraphState) -> GraphState:
        """
        Node 1: Retrieve documents from the vector store based on the question.
        """
        logger.info("---NODE: RETRIEVE---")
        question = state["question"]
        
        query_embedding = self.processor.encode_query(question)
        documents = self.vector_store.similarity_search(query_embedding, k=15)
        
        logger.info(f"Retrieved {len(documents)} documents.")
        return {**state, "documents": documents}


    
    ### CONDITIONAL EDGE FUNCTIONS ###
    def route_after_classification(self, state: GraphState) -> str:
        """
        Determines the next step based on input classification.
        """
        logger.info("---CONDITIONAL EDGE: ROUTE AFTER CLASSIFICATION---")
        classification = state.get("classification")
        # Use exact matching now that the classification is normalized and guaranteed to be one of the three categories.
        classification = state.get("classification", "")
        
        if classification == "off_topic_question":
            logger.info("Decision: Routing to off-topic response.")
            return "off_topic_response"
        elif classification == "conversational":
            logger.info("Decision: Routing to conversational response.")
            return "conversational_response"
        else: # on_topic_question
            is_list_request = state.get("is_list_request", False)
            if is_list_request:
                logger.info("Decision: On-topic list request. Routing to find_and_list_courses.")
                return "find_and_list_courses"
            else:
                logger.info("Decision: On-topic standard query. Routing to retrieve.")
                return "retrieve"

    def check_retrieval(self, state: GraphState) -> str:
        """
        After retrieval, check if any documents were found.
        """
        logger.info("---CONDITIONAL EDGE: CHECK RETRIEVAL---")
        if state.get("documents") and len(state["documents"]) > 0:
            logger.info("Decision: Documents found. Proceeding to format.")
            return "format_docs"
        else:
            logger.info("Decision: No documents found. Proceeding to fallback.")
            return "no_docs_fallback"
        

    def format_docs(self, state: GraphState) -> GraphState:
        """
        Node 4: Format the graded documents into a context string.
        This node leverages the robust formatting logic from the OpenRouterRAG class.
        """
        logger.info("---NODE: FORMAT DOCUMENTS---")
        documents = state["documents"]
        
        # Use the formatting function from our RAG model class
        context, sources = self.rag_model.format_context_with_sources(documents)
        
        return {**state, "context": context, "sources": sources}

    def generate(self, state: GraphState) -> GraphState:
        """
        Node 3: Generate an answer using the LLM based on the context.
        """
        logger.info("---NODE: GENERATE (Standard RAG Path)---")
        question = state["question"]
        context = state["context"]
        history = state.get("history", []) 
        # For the standard RAG path, we always use the general-purpose prompt.
        # The specialist curriculum prompt is handled by the `find_and_list_courses` node.
        answer = self.rag_model.query(
            prompt=question, 
            context=context, 
            history=history, 
            is_curriculum_question=False # This path is for non-curriculum list questions
        )
        
        logger.info(f"Generated answer: {answer[:100]}...")
        return {**state, "answer": answer}
    

    def no_docs_fallback(self, state: GraphState) -> GraphState:
        """
        A fallback node that provides a standard response when no documents are found.
        """
        logger.info("---NODE: NO DOCS FALLBACK---")
        answer = "I'm sorry, but I couldn't find any relevant information in the knowledge base to answer your question."
        return {**state, "answer": answer, "sources": []}
    
    def evaluate_answer(self, state: GraphState) -> GraphState:
        """
        Node: Evaluate the generated answer for faithfulness and relevancy.
        This uses an LLM-as-a-judge pattern and prints the results to the terminal.
        """
        logger.info("---NODE: EVALUATE ANSWER---")
        question = state["question"]
        context = state.get("context", "")
        answer = state.get("answer", "")

        # Skip evaluation for fallback messages or empty answers
        if not context or not answer or "I'm sorry" in answer:
            logger.info("Skipping evaluation for fallback or empty answers.")
            return {**state, "evaluation_results": {}}

        # 1. Faithfulness Check (Score 1-5)
        faithfulness_prompt = f"""You are an expert evaluator. Your task is to evaluate the "Generated Answer" for its faithfulness to the provided "Context" on a scale of 1 to 5.

- **1:** The answer is completely unfaithful and contradicts the context.
- **2:** The answer contains significant hallucinations or information not present in the context.
- **3:** The answer is mostly faithful but contains minor inaccuracies or additions.
- **4:** The answer is almost entirely faithful with only trivial deviations.
- **5:** The answer is perfectly faithful and all claims are directly verifiable from the context.

Context:
{context}

Generated Answer:
{answer}

Provide your evaluation in the following format:
Score: [Your score from 1 to 5]
Justification: [A brief explanation for your score]"""
        faithfulness_response = self.rag_model.query(prompt=faithfulness_prompt)
        faithfulness_score, faithfulness_justification = self._parse_evaluation(faithfulness_response)

        # 2. Answer Relevancy Check (Score 1-5)
        relevancy_prompt = f"""You are an expert evaluator. Your task is to evaluate the "Generated Answer" for its relevance to the "User's Question" on a scale of 1 to 5.

- **1:** The answer is completely irrelevant to the question.
- **2:** The answer is only slightly relevant and misses the main point of the question.
- **3:** The answer is partially relevant but incomplete or not fully satisfying.
- **4:** The answer is highly relevant and addresses most aspects of the question.
- **5:** The answer is perfectly relevant, direct, and completely addresses the user's question.

User's Question:
{question}

Generated Answer:
{answer}

Provide your evaluation in the following format:
Score: [Your score from 1 to 5]
Justification: [A brief explanation for your score]"""
        relevancy_response = self.rag_model.query(prompt=relevancy_prompt)
        relevancy_score, relevancy_justification = self._parse_evaluation(relevancy_response)

        evaluation_results = {
            "faithfulness_score": faithfulness_score,
            "faithfulness_justification": faithfulness_justification,
            "relevancy_score": relevancy_score,
            "relevancy_justification": relevancy_justification,
        }

        return {**state, "evaluation_results": evaluation_results}


    ### GRAPH BUILDER ###
    def build_graph(self):
        """
        Builds and compiles the LangGraph workflow.
        """
        workflow = StateGraph(GraphState)

        # Add the nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("conversational_response", self.conversational_response)
        workflow.add_node("off_topic_response", self.off_topic_response)
        workflow.add_node("find_and_list_courses", self.find_and_list_courses)    
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("format_docs", self.format_docs)        
        workflow.add_node("generate", self.generate)
        workflow.add_node("no_docs_fallback", self.no_docs_fallback)
        workflow.add_node("evaluate_answer", self.evaluate_answer)


        # Set the entry point
        workflow.set_entry_point("analyze_query")

        # Add the edges
        workflow.add_conditional_edges(
            "analyze_query",
            self.route_after_classification,
            {
                "find_and_list_courses": "find_and_list_courses",
                "retrieve": "retrieve",
                "conversational_response": "conversational_response",
                "off_topic_response": "off_topic_response",
            },
        )
       

        workflow.add_edge("conversational_response", END)
        workflow.add_edge("off_topic_response", END)

        
        workflow.add_edge("find_and_list_courses", "evaluate_answer")

        workflow.add_conditional_edges(
            "retrieve",
            self.check_retrieval,
            {
                "format_docs": "format_docs",
                "no_docs_fallback": "no_docs_fallback",
            },
        )

        # The main RAG pipeline
        workflow.add_edge("format_docs", "generate")
        workflow.add_edge("generate", "evaluate_answer")
        workflow.add_edge("no_docs_fallback", "evaluate_answer") # Also evaluate fallback to check relevancy
        workflow.add_edge("evaluate_answer", END) # Final step

        # Compile the graph
        logger.info("Compiling the RAG workflow graph.")
        return workflow.compile()

    def run(self, question: str, history: List[Dict] = None):
        """
        Runs the compiled graph with a user's question.
        """
        if history is None:
            history = []
            
        inputs = {"question": question, "history": history, "context": "", "documents": [], "answer": "", "sources": [], "classification": "", "is_curriculum_question": False, "evaluation_results": {}, "is_list_request": False}
        # The invoke method runs the graph from the entry point to an end point
        final_state = self.workflow.invoke(inputs)
        # Print the evaluation results to the terminal after the run is complete
        if final_state.get("evaluation_results"):
            results = final_state["evaluation_results"]
            if "faithfulness_score" in results:
                print("\n--- EVALUATION RESULTS ---")
                print(f"Faithfulness Score: {results['faithfulness_score']}/5")
                print(f"  Justification: {results['faithfulness_justification']}")
                print(f"Answer Relevancy Score: {results['relevancy_score']}/5")
                print(f"  Justification: {results['relevancy_justification']}")
                print("--------------------------\n")

        
        return final_state

    def stream(self, question: str, history: List[Dict] = None):
        """
        Streams the graph execution, yielding the output of each node as it completes.
        This is designed for interactive applications like Streamlit to show progress and provide faster feedback.
        """
        if history is None:
            history = []
            
        inputs = {"question": question, "history": history}
        
        # The stream method yields the output of each node as it's executed
        for output in self.workflow.stream(inputs):
            # The output is a dictionary with a single key: the name of the node that just ran.
            # The value is the dictionary of the state variables that were updated by that node.
            yield output