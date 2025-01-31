from dotenv import load_dotenv
import os
import logging
import time
import pandas as pd
import tiktoken
# Import OpenAI, Anthropic, and Gemini SDKs
import openai
import anthropic
import google.generativeai as genai
# Import evaluation tools
import json
from nltk.tokenize import sent_tokenize
import textstat
import nltk
# Import LaunchDarkly AI SDK
import ldclient
from ldclient import Context
from ldclient.config import Config
from ldai.client import LDAIClient, AIConfig, ModelConfig, LDMessage, ProviderConfig

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize LaunchDarkly client
ldclient.set_config(Config(os.getenv("LAUNCHDARKLY_SDK_KEY")))
ld_ai_client = LDAIClient(ldclient.get())

# Initialize model providers
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#gemini_client = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Google Generative AI client
gemini_client = os.getenv("GEMINI_API_KEY")
# Test the API client initialization
try:
    genai.configure(api_key=gemini_client)
except Exception as e:
    print(f"Failed to initialize Google Generative AI client: {e}")

# Generate function
def generate(context, model_name, messages, **kwargs):
    try:
        # Final placeholder replacement:
        name_placeholder = context.get("NAME") or "Default Name"
        final_messages = []
        for msg in messages:
            updated_content = msg["content"].replace("{{ ldctx.NAME }}", name_placeholder)
            final_messages.append({"role": msg["role"], "content": updated_content})

        logging.info(f"Final messages for {model_name}: {final_messages}")

        response = None
        if model_name.startswith("gpt"):
            response = query_openai(model_name, final_messages)
        elif model_name.startswith("claude"):
            response = query_claude(model_name, final_messages)
        elif model_name.startswith("gemini"):
            response = query_gemini(model_name, final_messages)
        else:
            return f"Error: Unknown model {model_name}"

        return response

    except Exception as e:
        logging.error(f"Error in generate: {e}")
        return f"Error: {e}"
    
# Query Antrhopic's Calude API using the official SDK
def query_claude(model_name, messages):
    try:
       # Extract system message if present
        system_message = next((msg.content.strip() for msg in messages if msg.role == "system"), None)

        
        # Prepare the messages list for the API request
        api_messages = [
            {"role": msg.role, "content": msg.content.strip()}
            for msg in messages if msg.role in ["user", "assistant"]
        ]

        logging.info(f"Query Claude with Messages: {api_messages}")
        
        # Track exeuction time
        start_time = time.time()
        
        # Call Claude's API
        response = anthropic_client.messages.create(
            model=model_name,
            system=system_message,
            messages=api_messages,
            max_tokens=1000,
            temperature=0.7,
        )
        elapsed_time = time.time() - start_time
        

        # Validate response
        if not response or not hasattr(response, "content") or not response.content:
            logging.error(f"Invalid response from Claude: {response}")
            return "Error: Claude returned an empty or invalid response."      

        # Extract the assistant's reply
        assistant_reply = "".join(
            block.text for block in response.content if block.type == "text"
        )
        
        # Log metrics
        tokens_used = len(assistant_reply.split()) * 100 # Approximate token usage per message
        cost = calculate_cost(model_name, tokens_used)
        log_metrics(model_name, assistant_reply, elapsed_time, tokens_used, cost)
        
        return assistant_reply
    
    except Exception as e:
        print(f"Error querying Claude: {e}")
        return f"Error: Unable to process the request with Claude: {e}"

# Gemini API Query
def query_gemini(model_name, messages):
    try:
        # Combine user messages into a single prompt
        prompt = "\n".join(msg.content for msg in messages if msg.role == "user")
        logging.info(f"Query Gemini with Prompt: {prompt}")

        # Initialize the Gemini model dynamically
        model = genai.GenerativeModel(model_name)

        # Track execution time
        start_time = time.time()

        # Call Gemini's API
        response = model.generate_content(prompt)

        # Calculate cost
        elapsed_time = time.time() - start_time
        tokens_used = len(prompt.split()) # Approximate token usage
        cost = calculate_cost(model_name, tokens_used)

        # Validate response
        if not response or not hasattr(response, "candidates") or not response.candidates:
            logging.error(f"Invalid response from Gemini: {response}")
            return "Error: Gemini returned an empty or invalid response."

        # Extract the generated text
        result = response.text

        # Log metrics
        log_metrics(model_name, result, elapsed_time, tokens_used, cost)
        return result
    
    except Exception as e:
        logging.error(f"Error querying Gemini: {e}")
        return f"Error: Unable to process the request with Gemini: {e}"

# OpenAI API Query
def query_openai(model_name, messages):
    try:
        start_time = time.time()
        completion = openai_client.chat.completions.create(
            model=model_name,
            # Just pass the dicts:
            messages=messages,
        )
        elapsed_time = time.time() - start_time

        tokens_used = completion.usage.total_tokens
        cost = calculate_cost(model_name, tokens_used)

        if not completion or not hasattr(completion, "choices") or not completion.choices:
            logging.error(f"Invalid response from OpenAI: {completion}")
            return "Error: OpenAI returned an empty or invalid response."

        response_content = completion.choices[0].message.content
        log_metrics(model_name, response_content, elapsed_time, tokens_used, cost)
        return response_content

    except Exception as e:
        logging.error(f"Error querying OpenAI: {e}")
        return f"Error: Unable to process the request with OpenAI: {e}"
    
# Calculate cost
def calculate_cost(model_name, tokens):
    cost_map = {
        "gpt-4o": 0.03,
        "chat-gpt-4o-latest": 0.04,
        "claude-3-5-haiku-20241022": 0.025,
        "gemini-1.5-flash-002": 0.02,
    }
    return cost_map.get(model_name, 0.0) * tokens

# Logs performance metrics for the model's response
def log_metrics(model_name, response, elapsed_time, tokens_used, cost):
    logging.info(f"Model: {model_name}")
    #logging.info(f"Response: {response}")
    logging.info(f"Execution time: {elapsed_time:.2f} seconds")
    logging.info(f"Tokens used: {tokens_used}")
    logging.info(f"Cost: ${cost:.2f}")


def validate_api_keys():
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LAUNCHDARKLY_SDK_KEY"]
    for key in required_keys:
        if not os.getenv(key):
            raise ValueError(f"Missing required API key: {key}")
    logging.info("API keys are valid")


# Calculate the number of tokens used by a given set of messages based on the model
def calculate_tokens(model_name, messages):
    if model_name.startswith("gpt"):
        return calculate_openai_tokens(model_name, messages)
    elif model_name.startswith("claude"):
        return calculate_anthropic_tokens(model_name, messages)   # <-- pass model_name!
    elif model_name.startswith("gemini"):
        return calculate_gemini_tokens(messages)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Calculate the number of tokens used by OpenAI
def calculate_openai_tokens(model_name, messages):
    encoding = tiktoken.encoding_for_model(model_name)
    token_count = 0
    for msg in messages:
        # Now msg is always a dictionary with "role" and "content"
        content = msg["content"]
        role = msg["role"]
        token_count += len(encoding.encode(content)) + len(encoding.encode(role))
    return token_count

def calculate_anthropic_tokens(model_name, messages):
    # messages is a list of dicts, so use msg["content"]
    return sum(len(msg["content"].split()) for msg in messages) + (len(messages) * 3)

# Calculate the number of tokens used by Gemini
def calculate_gemini_tokens(messages):
    # Approximate: 1 token = 4 characters
    char_count = sum(len(msg.content) for msg in messages if hasattr(msg, 'content'))
    return char_count // 4


# Fetch all variations from LaunchDarkly and test their outputs
def fetch_all_variations_from_launchdarkly(context):
    try:
        # Fetch all flags and filter for the specific AI Config
        flag_state = ldclient.get().all_flags_state(context)
        all_flags = flag_state.to_values_map()

        # Retrieve only the AI Config key
        ai_config = all_flags.get("model-upgrade", {})
        logging.info(f"Fetched AI Config: {ai_config}")
        return ai_config

    except Exception as e:
        logging.error(f"Error fetching AI Config: {e}")
        return {}


# Generate for models using LaunchDarkly
def generate_for_models_using_launchdarkly(context, **kwargs):
    results = []
    try:
        logging.info("Fetching all variations from LaunchDarkly...")
        ai_config = fetch_all_variations_from_launchdarkly(context)

        if not ai_config or not isinstance(ai_config, dict):
            logging.error("No AI Config variations found in LaunchDarkly.")
            return results

        # Process the AI Config
        try:
            logging.info(f"Testing AI Config: {ai_config}")

            # Extract model details
            model_data = ai_config.get("model")
            if not model_data or "name" not in model_data:
                logging.error("Skipping: Missing required model data.")
                return results

            model_name = model_data["name"]
            
            # Extract messages
            messages = ai_config.get("messages", [])
            if not messages:
                logging.error("Skipping: Missing messages in AI Config.")
                return results

            # Retrieve NAME attribute from context
            name_placeholder = context.get("NAME") or "Default Name"

            
            # Process messages correctly
            processed_messages = [
                to_dict_message(msg, name_placeholder)
                for msg in messages
            ]

            # Measure latency and get response
            start_time = time.time()
            response = generate(context, model_name=model_name, messages=processed_messages, **kwargs)
            elapsed_time = time.time() - start_time

            tokens_used = calculate_tokens(model_name, processed_messages)
            cost = calculate_cost(model_name, tokens_used)
            
            # Get evaluation results
            hybrid_eval = hybrid_evaluation(response)

            # Log and store results
            results.append({
                "model": model_name,
                "variation": ai_config.get("_ldMeta", {}).get("variationKey", "unknown"),
                "response": response,
                "latency": elapsed_time,
                "tokens_used": tokens_used,
                "cost": cost,
                "evaluation": hybrid_eval
            })

        except Exception as e:
            logging.error(f"Error processing AI Config: {e}")

    except Exception as e:
        logging.error(f"Error fetching variations or generating for models: {e}")

    return results

# Output results in a tabular format for analysis
def compare_model_outputs(results):
    """
    Enhanced comparison of model outputs including detailed evaluation metrics
    """
    # Create a flattened version of results for DataFrame
    flattened_results = []
    
    print("\n=== Detailed Model Evaluation Results ===\n")
    
    for result in results:
        # Print the detailed response and evaluation for each result
        print(f"\nModel: {result.get('model', 'unknown')}")
        print(f"Variation: {result.get('variation', 'unknown')}")
        print("\nResponse:")
        print("-" * 80)
        print(result.get('response', 'No response available'))
        print("-" * 80)
        
        # Get evaluation data
        evaluation = result.get("evaluation", {})
        gpt4_analysis = evaluation.get("gpt4_analysis", {})
        textstat_analysis = evaluation.get("textstat_analysis", {})
        
        # Print GPT-4 Analysis
        print("\nGPT-4 Evaluation:")
        print(f"Professional Tone: {gpt4_analysis.get('professional_tone', 0)}/5")
        print(f"Human-likeness: {gpt4_analysis.get('human_likeness', 0)}/5")
        print(f"Clarity & Readability: {gpt4_analysis.get('clarity_readability', 0)}/5")
        
        # Print detailed analysis if available
        analysis = gpt4_analysis.get('analysis', {})
        if analysis:
            print("\nStrengths:")
            for strength in analysis.get('strengths', []):
                print(f"- {strength}")
            print("\nAreas for Improvement:")
            for improvement in analysis.get('improvements', []):
                print(f"- {improvement}")
            print(f"\nEffectiveness: {analysis.get('effectiveness', 'Not available')}")
        
        # Print TextStat Analysis
        print("\nReadability Metrics:")
        readability_scores = textstat_analysis.get("readability_scores", {})
        print(f"Flesch Reading Ease: {readability_scores.get('flesch_score', 0):.2f}")
        print(f"Flesch-Kincaid Grade Level: {readability_scores.get('flesch_kincaid_grade', 0):.2f}")
        print(f"Gunning Fog Index: {readability_scores.get('gunning_fog', 0):.2f}")
        
        # Print combined metrics
        combined_metrics = evaluation.get("combined_metrics", {})
        print("\nCombined Metrics:")
        print(f"Overall Readability: {combined_metrics.get('overall_readability', 0):.2f}")
        print(f"Complexity Score: {combined_metrics.get('complexity_score', 0):.2f}")
        print(f"Professional Quality: {combined_metrics.get('professional_quality', 0):.2f}")
        print(f"Human-likeness: {combined_metrics.get('human_likeness', 0):.2f}")
        print(f"Education Level: {combined_metrics.get('education_level', 'Not available')}")
        
        print("\n" + "=" * 80 + "\n")
        
        # Create flattened result for DataFrame
        flat_result = {
            "model": result.get("model", "unknown"),
            "variation": result.get("variation", "unknown"),
            "latency": result.get("latency", 0),
            "tokens_used": result.get("tokens_used", 0),
            "cost": result.get("cost", 0),
            "response_length": len(result.get("response", "")),
            "professional_tone": gpt4_analysis.get("professional_tone", 0),
            "human_likeness": gpt4_analysis.get("human_likeness", 0),
            "clarity_readability": gpt4_analysis.get("clarity_readability", 0),
            "flesch_score": readability_scores.get("flesch_score", 0),
            "grade_level": readability_scores.get("flesch_kincaid_grade", 0),
            "fog_index": readability_scores.get("gunning_fog", 0)
        }
        flattened_results.append(flat_result)
    
    try:
        # Create DataFrame
        df = pd.DataFrame(flattened_results)
        
        # Add summary statistics
        summary_stats = df.describe()
        
        # Save detailed results
        df.to_csv("model_comparisons_detailed.csv", index=False)
        summary_stats.to_csv("model_comparisons_summary.csv")
        
        # Print summary
        print("\nStatistical Summary:")
        print(summary_stats)
        
        logging.info("Detailed model comparisons saved to model_comparisons_detailed.csv")
        logging.info("Summary statistics saved to model_comparisons_summary.csv")
    except Exception as e:
        logging.error(f"Error in creating comparison outputs: {e}")
        print("Failed to create comparison outputs. Check the logs for details.")

# Convert LDMessage or dictionary into a standard dictionary with role and content, replacing {{ ldctx.NAME }} placeholders.
def to_dict_message(msg, name_placeholder="Default Name"):
    """
    Converts an LDMessage or dictionary into a standard dictionary
    with role and content, replacing {{ ldctx.NAME }} placeholders.
    """
    if isinstance(msg, LDMessage):
        # If it's an LDMessage, use msg.role / msg.content
        return {
            "role": msg.role,
            "content": msg.content.replace("{{ ldctx.NAME }}", name_placeholder)
        }
    elif isinstance(msg, dict):
        # If it's already a dict, just ensure content is replaced
        new_content = msg.get("content", "").replace("{{ ldctx.NAME }}", name_placeholder)
        return {
            "role": msg.get("role", "user"),
            "content": new_content
        }
    else:
        # Fallback in case some unknown format appears
        return {"role": "user", "content": str(msg)}

# Main function to test multiple models
def main():
    logging.info("Testing multiple models using LaunchDarkly...")

    # Create a context for the user
    context = Context.builder("context-key-123abc").anonymous(True).set("NAME", "Stephen Curry").build()

    # Fetch and test all variations for AI Configs
    results = generate_for_models_using_launchdarkly(context)

    # Check if results are empty
    if not results:
        logging.error("No results were generated. Ensure AI Config variations are properly set in LaunchDarkly.")
        return

    # Print results
    for result in results:
        logging.info(
            f"Model: {result['model']}, Variation: {result['variation']}, "
            f"Response: {result['response']}, Latency: {result['latency']:.2f} seconds, "
            f"Tokens Used: {result['tokens_used']}, Cost: ${result['cost']:.2f}"
        )

    # Analyze and save the results
    compare_model_outputs(results)

def evaluate_response_with_gpt4(response):
    """
    Uses GPT-4 to evaluate response quality focusing on professional metrics.
    """
    evaluation_prompt = f"""
    Please evaluate the following AI response and return a JSON object with these specific criteria:
    
    1. Professional Tone (0-5):
       - Formality level
       - Business appropriateness
       - Technical accuracy
    
    2. Human-likeness (0-5):
       - Natural language flow
       - Conversational elements
       - Emotional intelligence
    
    3. Clarity & Readability (0-5):
       - Message comprehension
       - Structure and organization
       - Conciseness
    
    4. Detailed Analysis:
       - Key strengths
       - Areas for improvement
       - Overall effectiveness
    
    Response to evaluate: "{response}"
    
    Format your response as valid JSON with these exact keys:
    {{
        "professional_tone": <score>,
        "human_likeness": <score>,
        "clarity_readability": <score>,
        "analysis": {{
            "strengths": [...],
            "improvements": [...],
            "effectiveness": "..."
        }}
    }}
    """
    
    try:
        evaluation_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional content evaluator. Always respond with valid JSON."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.7
        )
        
        # Extract the response content
        response_text = evaluation_response.choices[0].message.content
        
        try:
            # Parse the JSON response
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse GPT-4 response as JSON: {e}")
            # Return a default structure if JSON parsing fails
            return {
                "professional_tone": 0,
                "human_likeness": 0,
                "clarity_readability": 0,
                "analysis": {
                    "strengths": [],
                    "improvements": [],
                    "effectiveness": "Evaluation failed"
                }
            }
    except Exception as e:
        logging.error(f"Error in GPT-4 evaluation: {e}")
        # Return a default structure if the API call fails
        return {
            "professional_tone": 0,
            "human_likeness": 0,
            "clarity_readability": 0,
            "analysis": {
                "strengths": [],
                "improvements": [],
                "effectiveness": f"Evaluation failed: {str(e)}"
            }
        }

def evaluate_response_with_textstat(response):
    """
    Uses TextStat to evaluate response readability and complexity.
    """
    try:
        evaluation = {
            "readability_scores": {
                "flesch_score": textstat.flesch_reading_ease(response),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(response),
                "gunning_fog": textstat.gunning_fog(response),
                "smog_index": textstat.smog_index(response),
                "automated_readability": textstat.automated_readability_index(response),
                "coleman_liau": textstat.coleman_liau_index(response),
                "dale_chall": textstat.dale_chall_readability_score(response)
            },
            "text_metrics": {
                "syllable_count": textstat.syllable_count(response),
                "lexicon_count": textstat.lexicon_count(response),
                "sentence_count": textstat.sentence_count(response),
                "difficult_words": textstat.difficult_words(response),
                "text_standard": textstat.text_standard(response)
            }
        }
        return evaluation
    except Exception as e:
        logging.error(f"Error in TextStat evaluation: {e}")
        return None

def hybrid_evaluation(response):
    try:
        gpt_eval = evaluate_response_with_gpt4(response)
        textstat_eval = evaluate_response_with_textstat(response)
        
        hybrid_score = {
            "gpt4_analysis": gpt_eval,
            "textstat_analysis": textstat_eval,
            "combined_metrics": {}
        }
        
        if gpt_eval and textstat_eval:
            try:
                flesch_normalized = textstat_eval["readability_scores"]["flesch_score"] / 20
                clarity_score = gpt_eval.get("clarity_readability", 0)
                
                hybrid_score["combined_metrics"] = {
                    "overall_readability": (flesch_normalized + clarity_score) / 2,
                    "complexity_score": textstat_eval["readability_scores"]["gunning_fog"] / 20,
                    "professional_quality": gpt_eval.get("professional_tone", 0),
                    "human_likeness": gpt_eval.get("human_likeness", 0),
                    "education_level": textstat_eval["text_metrics"]["text_standard"]
                }
            except KeyError as e:
                logging.error(f"Missing expected key in evaluation results: {e}")
            except Exception as e:
                logging.error(f"Error calculating combined metrics: {e}")
        
        return hybrid_score
    except Exception as e:
        logging.error(f"Error in hybrid evaluation: {e}")
        return None

def check_dependencies():
    """
    Verify all required dependencies and environment variables are available.
    """
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    required_env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "LAUNCHDARKLY_SDK_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")

if __name__ == "__main__":
    check_dependencies()
    main()