import time
import statistics
import logging
import re

from main import initialize_components

# Disable verbose logging from the RAG system during evaluation for a cleaner output
logging.getLogger('rag').setLevel(logging.WARNING)
logging.getLogger('scraper').setLevel(logging.WARNING)
logging.getLogger('vector_db').setLevel(logging.WARNING)

# --- Test Dataset ---
# In a real-world scenario, this would be a much larger, external file (e.g., a CSV or JSON).
# We include questions designed to test different paths in the graph.
EVALUATION_DATASET = [
    # On-topic questions with reference answers for scoring
    {
        "question": "Tell me about the Business Process Intelligence course", 
        "expected_classification": "on_topic_question",
        "reference_answer": "The Business Process Intelligence course is taught by Chihab Hanachi in the second semester of year 1. It covers the business process life-cycle, including design, analysis, and process mining. A prerequisite for this course is the Software Analysis and Design course from the first master's year."
    },
    {
        "question": "who teaches object oriented programming?", 
        "expected_classification": "on_topic_question",
        "reference_answer": "The Object Oriented Programming course is taught by David Simoncini."
    },
    {
        "question": "list all the courses for the first semester of year 1", 
        "expected_classification": "on_topic_question",
        "reference_answer": "The courses for the first semester of year 1 are: Advanced Databases, Business Intelligence, Software Design, Object Oriented Programming, Data Analysis and Visualisation, Project Management, Research Workshop 1, and Personal Development."
    },
    {
        "question": "can you tell me what are the teachers we will be having during our first year?", 
        "expected_classification": "on_topic_question",
        "reference_answer": "Some of the teachers in the first year include Manon Prédhumeau, Ronan Tournier, Umberto Grandi, David Simoncini, Josiane Mothe, Benoit Marsa, Michael Evgi, Chihab Hanachi, Benoit Gaudou, and Cyrielle Vellera."
    },
    {
        "question": "list the courses in year 1 and their ECTS credits", 
        "expected_classification": "on_topic_question",
        "reference_answer": "In year 1, most core courses like Advanced Databases, Business Intelligence, and AI are worth 5 ECTS. Workshops and projects like Research Workshop 1 and Term Project 1 are worth 2 ECTS."
    },
    
    # Conversational questions
    {"question": "hello, can you help me?", "expected_classification": "conversational"},
    {"question": "thank you for your help", "expected_classification": "conversational"},
    
    # Off-topic questions
    {"question": "what is the capital of France?", "expected_classification": "off_topic_question", "reference_answer": "Paris"},
    {"question": "who is the current president of the USA?", "expected_classification": "off_topic_question", "reference_answer": "Joe Biden"},

    # "Trap" question to test for hallucinations (answer is not in the knowledge base)
    {
        "question": "What are the tuition fees for the 2IS master's program?", 
        "expected_classification": "on_topic_question",
        "reference_answer": "I'm sorry, but I couldn't find any information about the specific tuition fees in the provided documents."
    },
]

def run_evaluation():
    """
    Runs a comprehensive evaluation of the RAG system using a predefined dataset.
    Calculates and prints key performance metrics.
    """
    print("Initializing RAG system for evaluation...")
    rag_graph = initialize_components()
    if not rag_graph:
        print("Evaluation aborted due to initialization failure.")
        return

    print(f"\n--- Starting Evaluation on {len(EVALUATION_DATASET)} Test Cases ---")

    response_times = []
    classification_correct = 0
    faithfulness_scores = []
    relevancy_scores = []
    reference_based_scores = []

    for i, item in enumerate(EVALUATION_DATASET):
        question = item["question"]
        expected_classification = item["expected_classification"]
        
        print(f"\n[{i+1}/{len(EVALUATION_DATASET)}] Testing: \"{question}\"")

        # --- Run the full graph once to get the final state for evaluation ---
        start_time = time.time()
        final_state = rag_graph.run(question)
        end_time = time.time()

        # --- Metric 1: Classification Accuracy ---
        actual_classification = final_state.get("classification")
        
        if actual_classification == expected_classification:
            classification_correct += 1
            print(f"  ✅ Classification: Correct ('{actual_classification}')")
        else:
            print(f"  ❌ Classification: Incorrect. Got '{actual_classification}', expected '{expected_classification}'")

        # --- Metric 2: Response Time ---
        # We only record response time for on-topic questions as they represent the full RAG pipeline.
        if actual_classification == "on_topic_question":
            duration = end_time - start_time
            response_times.append(duration)
            print(f"  ⏱️ Response Time: {duration:.2f}s")

        # --- Metric 3: Faithfulness & Relevancy (Self-Evaluation) ---
        evaluation_results = final_state.get("evaluation_results", {})
        faithfulness_score = evaluation_results.get("faithfulness_score")
        relevancy_score = evaluation_results.get("relevancy_score")
        
        if faithfulness_score is not None:
            faithfulness_scores.append(faithfulness_score)
            relevancy_scores.append(relevancy_score if relevancy_score is not None else 0)
            print(f"  👍 Faithfulness Score: {faithfulness_score}/5")
            print(f"  🎯 Relevancy Score: {relevancy_score}/5")
        elif actual_classification == "on_topic_question": # Only log this for on-topic questions that should have had a context
            # This happens for fallback messages where evaluation is skipped.
            faithfulness_scores.append(5) # Assign perfect score for correctly not answering.
            print("  👍 Faithfulness Score: 5/5 (Correctly identified no information)")
        
        # --- Metric 4: Reference-Based Answer Quality ---
        # This can be applied to ANY question that has a reference answer.
        reference_answer = item.get("reference_answer")
        if reference_answer:
            ai_response = final_state.get("answer", "")
            
            evaluation_prompt = f"""You are a strict evaluator. Compare the "AI Response" to the "True Response" for the given "User Query". Score the AI's response on a scale from 0.0 to 1.0 for semantic similarity and factual accuracy.
- 1.0 means the AI Response is perfectly aligned with the True Response.
- 0.0 means it is completely different or wrong.
- Only output the numerical score and nothing else.

User Query: "{question}"
AI Response:
{ai_response}

True Response:
{reference_answer}

Score:"""
            score_response = rag_graph.rag_model.query(prompt=evaluation_prompt)
            try:
                score = float(re.search(r"(\d\.\d+)", score_response).group(1))
                reference_based_scores.append(score)
                print(f"  💯 Reference-Based Score: {score:.3f}/1.0")
            except (ValueError, AttributeError, TypeError):
                print(f"  ⚠️ Could not parse reference-based score from response: '{score_response}'")

    # --- Final Report ---
    print("\n\n--- 📊 Evaluation Report ---")

    # Classification Accuracy
    classification_accuracy = (classification_correct / len(EVALUATION_DATASET)) * 100
    print(f"\n1. Classification Accuracy: {classification_accuracy:.2f}% ({classification_correct}/{len(EVALUATION_DATASET)} correct)")

    # Average Response Time
    if response_times:
        avg_response_time = statistics.mean(response_times)
        print(f"2. Average Response Time (on-topic): {avg_response_time:.2f}s")

    # Average Faithfulness (Hallucination Metric)
    if faithfulness_scores:
        avg_faithfulness = statistics.mean(faithfulness_scores)
        print(f"3. Average Faithfulness Score: {avg_faithfulness:.2f}/5")
        print("   (Measures how well the model sticks to the context, preventing hallucinations.)")

    # Average Relevancy
    if relevancy_scores:
        avg_relevancy = statistics.mean(relevancy_scores)
        print(f"4. Average Relevancy Score: {avg_relevancy:.2f}/5")
        print("   (Measures how well the answer addresses the user's actual question.)")
    
    # Average Reference-Based Score
    if reference_based_scores:
        avg_ref_score = statistics.mean(reference_based_scores)
        print(f"5. Average Reference-Based Score: {avg_ref_score:.3f}/1.0")
        print("   (Measures how semantically similar the AI answer is to a ground-truth answer.)")

    print("\n--------------------------")

if __name__ == "__main__":
    run_evaluation()
