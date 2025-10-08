#main_ds.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
import json
import uvicorn
from pathlib import Path
import os
import glob
from typing import List, Optional, Dict, Any
import boto3
from io import BytesIO
import yaml
from datetime import datetime

# --- Enhanced Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Smart model mapping for different tasks
OLLAMA_MODELS = {
    "sql": "sqlcoder",           # SQL generation and optimization
    "dbt": "sqlcoder",           # dbt tests and YAML
    "quality": "sqlcoder",       # Data quality checks
    "pipeline": "codellama",     # Python pipeline code
    "s3": "codellama",           # Python S3 operations
    "chat": "gemma3:1b",         # General conversation and explanations
    "documentation": "gemma3:1b", # Documentation writing
    "schema_mapper": "sqlcoder",  # Manual schema mapping (renamed from schema_harmonizer)
    "schema_harmonizer": "sqlcoder",  # File-based schema harmonization
    "default": "sqlcoder"        # Fallback model
}

app = FastAPI(
    title="DataGenie AI Co-Pilot",
    description="AI-powered data engineering assistant with SQL generation, pipeline design, and data quality tools.",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Enhanced Pydantic Models ---
class QAPair(BaseModel):
    question: str
    answer: str


class FileInfo(BaseModel):
    name: str
    path: str
    columns: List[str]

class ConfigInput(BaseModel):
    base_path: str
    files: List[dict]
    join_order: List[str]
    output_config_path: str
    filter_column: str  # Dynamic filter column

class QueryLog(BaseModel):
    prompt_type: str
    target_id: int
    prompt_used: str
    llm_response: str
    llm_model: str
    success: bool
    timestamp: str

class ReportResponse(BaseModel):
    report_id: int
    llm_model: str
    results: List[QAPair]
    llm_generation_success: bool = True
    query_log: Optional[QueryLog] = None  # Add this field
    

# --- AI Module Request Models ---
class SQLRequest(BaseModel):
    prompt: str
    schema_context: str = ""
    database_type: str = "snowflake"

class DBTRequest(BaseModel):
    table_name: str
    columns: List[str]
    requirements: str = ""

class PipelineRequest(BaseModel):
    description: str
    source_type: str = "s3"
    target_type: str = "snowflake"
    tools: List[str] = ["airflow", "dbt"]

class S3Request(BaseModel):
    operation: str
    bucket: str
    path: str = ""
    file_format: str = "parquet"

class QualityRequest(BaseModel):
    dataset: str
    checks: str
    framework: str = "dbt"

class ChatRequest(BaseModel):
    message: str
    context: str = ""

# --- Schema Mapper (Manual Schema Mapping) ---
class SchemaMapperRequest(BaseModel):
    source_schema: Dict[str, str]  # {"column_name": "data_type"}
    target_schema: Dict[str, str]  # {"column_name": "data_type"} 
    source_system: str = "unknown"
    target_system: str = "unknown"
    mapping_rules: str = ""

# --- Schema Harmonizer (File-based with UI) ---
class SchemaHarmonizerRequest(BaseModel):
    config: dict  # The config from UI (base_path, files, join_order, etc.)
    target_system: str = "snowflake"
    mapping_requirements: str = ""
    output_config_path: str = ""  # Where to save the harmonized config

# --- AI Prompt Templates ---
AI_PROMPTS = {
    "sql_generation": """As an expert data engineer, generate optimized {database_type} SQL for this request:

REQUEST: {prompt}

SCHEMA CONTEXT:
{schema_context}

Requirements:
- Use proper SQL syntax for {database_type}
- Include appropriate joins and filters
- Optimize for performance
- Handle edge cases and NULL values
- Add comments for complex logic

Return only the SQL query without explanations:""",

    "dbt_tests": """Create comprehensive dbt data quality tests for table '{table_name}' with columns: {columns}

SPECIFIC REQUIREMENTS:
{requirements}

Generate a complete dbt schema.yml file with:
- Column descriptions
- Data quality tests (unique, not_null, relationships)
- Custom tests for business logic
- Appropriate test severity levels

Return valid YAML only:""",

    "pipeline_design": """Design a data pipeline with these specifications:

DESCRIPTION: {description}
SOURCE: {source_type}
TARGET: {target_type}
TOOLS: {tools}

Create a complete pipeline implementation including:
1. Data extraction logic
2. Transformation steps
3. Loading strategy
4. Error handling
5. Monitoring and logging

Return production-ready Python code:""",

    "data_quality": """Generate data quality checks for dataset: {dataset}

QUALITY CHECKS NEEDED:
{checks}

FRAMEWORK: {framework}

Create comprehensive quality checks covering:
- Completeness and validity
- Uniqueness and consistency
- Accuracy and timeliness
- Business rule validation

Return executable quality checks:""",

    "s3_operations": """Generate Python code for S3 {operation} operation:

BUCKET: {bucket}
PATH: {path}
FORMAT: {file_format}

Requirements:
- Use boto3 library
- Include proper error handling
- Handle different file formats appropriately
- Include documentation comments

Return complete, runnable Python code:""",

    "schema_mapper": """Analyze and map schemas between source and target systems:

SOURCE SYSTEM: {source_system}
TARGET SYSTEM: {target_system}

SOURCE SCHEMA:
{source_schema}

TARGET SCHEMA:  
{target_schema}

MAPPING RULES:
{mapping_rules}

Generate a comprehensive schema mapping plan including:
1. Column mapping recommendations
2. Data type conversions and compatibility
3. Transformation rules for mismatched columns
4. Data validation checks
5. Migration SQL scripts

Return a structured analysis with actionable recommendations:""",

    "schema_harmonizer": """Analyze this multi-table schema configuration and create a harmonized target schema:

TARGET SYSTEM: {target_system}

SOURCE CONFIGURATION:
{source_config}

MAPPING REQUIREMENTS:
{mapping_requirements}

Generate a comprehensive schema harmonization plan including:

1. HARMONIZED SCHEMA DESIGN:
   - Optimized table structure for {target_system}
   - Appropriate data types and constraints
   - Partitioning and clustering recommendations

2. COLUMN MAPPING & TRANSFORMATIONS:
   - Source to target column mappings
   - Data type conversions
   - Transformation logic for mismatched columns
   - Business rule implementations

3. MIGRATION STRATEGY:
   - DDL statements for target tables
   - ETL transformation logic
   - Data validation queries
   - Performance optimization

4. CONFIGURATION FILE:
   - Ready-to-use configuration for data pipelines
   - Table relationships and dependencies
   - Data quality rules

Return structured JSON with these sections."""
}
# --- Authentication Prompt Templates ---

AUTHENTICATION_PROMPTS = {
    "behavioral": """As a bank verification agent, analyze this customer data for {target_id} and generate exactly 3 natural, conversational behavioral verification questions that sound like what a real bank agent would ask during a phone call.

DATA:
{data_str}

CRITICAL REQUIREMENTS:
- Questions must be natural and conversational, not robotic
- Answers must contain SPECIFIC factual information extracted from the actual data
- Use ACTUAL VALUES from the data, not placeholders like [pattern details]
- Focus on spending patterns, transaction habits, and behavioral trends
- Make questions sound like genuine verification questions

Return ONLY valid JSON array with exactly 3 question-answer pairs:
[
  {{
    "question": "Natural behavioral question about spending patterns",
    "answer": "Specific factual answer using real values from the data"
  }},
  {{
    "question": "Another natural behavioral question about transaction habits",
    "answer": "Another specific answer with actual data values"
  }},
  {{
    "question": "Final behavioral verification question",
    "answer": "Final specific answer with real behavioral data"
  }}
]

IMPORTANT: Extract and use real behavioral patterns from the data. Do not use [bracketed placeholders].""",

    "knowledge": """As a bank verification agent, analyze this customer data for {target_id} and generate exactly 3 natural, conversational knowledge-based verification questions that sound like what a real bank agent would ask during identity verification.

DATA:
{data_str}

CRITICAL REQUIREMENTS:
- Questions must sound like genuine bank verification questions
- Answers must contain SPECIFIC factual information extracted from the actual data
- Use ACTUAL VALUES from the data, not placeholders like [specific details]
- Focus on account details, transaction history, and personal information
- Make questions varied and cover different aspects of the account

Examples of GOOD questions:
- "Can you verify the date of your last payment and the amount?"
- "What's the current balance showing on your main account?"
- "Which industry does our records show for your employment?"

Return ONLY valid JSON array with exactly 3 question-answer pairs:
[
  {{
    "question": "Natural knowledge question about account details",
    "answer": "Specific factual answer using real values from the data"
  }},
  {{
    "question": "Another natural knowledge question about transactions",
    "answer": "Another specific answer with actual data values"
  }},
  {{
    "question": "Final knowledge verification question",
    "answer": "Final specific answer with real account data"
  }}
]

IMPORTANT: Extract and use real values from the data. Do not use [bracketed placeholders].""",

    "multifactor": """As a bank verification agent, analyze this customer data for {target_id} and generate exactly 3 comprehensive multi-factor verification questions that combine behavioral patterns and account knowledge.

DATA:
{data_str}

CRITICAL REQUIREMENTS:
- Questions must combine behavioral patterns and specific account knowledge
- Answers must contain SPECIFIC factual information extracted from the actual data
- Use ACTUAL VALUES from the data, not placeholders like [specific steps]
- Questions should verify identity through multiple verification factors
- Make questions sound like comprehensive security verification

Return ONLY valid JSON array with exactly 3 question-answer pairs:
[
  {{
    "question": "Multi-factor question combining behavior and knowledge",
    "answer": "Specific factual answer using real values from the data"
  }},
  {{
    "question": "Another multi-factor verification question",
    "answer": "Another specific answer with actual data values"
  }},
  {{
    "question": "Final multi-factor security question",
    "answer": "Final specific answer with real verification data"
  }}
]

IMPORTANT: Extract and use real patterns and values from the data. Do not use [bracketed placeholders].""",

    "security": """As a bank security specialist, analyze this customer data for {target_id} and generate exactly 3 security-focused verification questions that sound like what a security agent would ask during fraud prevention.

DATA:
{data_str}

CRITICAL REQUIREMENTS:
- Questions must focus on security patterns and fraud prevention
- Answers must contain SPECIFIC factual information extracted from the actual data
- Use ACTUAL VALUES from the data, not placeholders like [specific methods]
- Focus on transaction verification, account monitoring, and security habits
- Make questions sound like professional security verification

Return ONLY valid JSON array with exactly 3 question-answer pairs:
[
  {{
    "question": "Security-focused verification question",
    "answer": "Specific factual answer using real values from the data"
  }},
  {{
    "question": "Another security verification question",
    "answer": "Another specific answer with actual security data"
  }},
  {{
    "question": "Final security confirmation question",
    "answer": "Final specific answer with real security patterns"
  }}
]

IMPORTANT: Extract and use real security patterns from the data. Do not use [bracketed placeholders]."""
}
# --- Enhanced Helper Functions ---
def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ["path", "files", "join_order"]
        if not all(key in config for key in required_keys):
            raise ValueError(f"Config missing required keys: {required_keys}")
        
        return config
    except Exception as e:
        raise ValueError(f"Error loading config: {str(e)}")
    
def load_and_filter_data(national_id: int, config_path: str) -> pd.DataFrame:
    """Load and filter data for specific national ID - SIMPLER APPROACH"""
    try:
        print(f"🔧 Starting data load for ID: {national_id}, config: {config_path}")
        
        config = load_config(config_path)
        base_path = Path(config["path"])
        files = config["files"]
        join_order = config["join_order"]
        
        # Get the filter column from config
        filter_column = config.get("filter_column", "PersonNationalID")
        print(f"🔑 Using filter column: {filter_column}")

        # Load all CSV files
        dfs = {}
        for file_info in files:
            file_name = file_info["name"]
            file_path = base_path / file_info["path"]
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            dfs[file_name] = pd.read_csv(file_path)
            print(f"✅ Loaded {file_name}: {dfs[file_name].shape}")

        # Start with first file
        result_df = dfs[join_order[0]].copy()

        # Perform joins
        for i in range(1, len(join_order)):
            current_file = join_order[i]
            prev_file = join_order[i-1]
            
            current_info = next(f for f in files if f["name"] == current_file)
            prev_info = next(f for f in files if f["name"] == prev_file)
            
            join_key_in = current_info.get("join_key_in")
            join_key_out = prev_info.get("join_key_out")
            
            if join_key_in and join_key_out:
                # Handle column name conflicts
                current_df = dfs[current_file].copy()
                for col in current_df.columns:
                    if col in result_df.columns and col != join_key_in:
                        current_df = current_df.rename(columns={col: f"{col}_{current_file}"})
                
                result_df = result_df.merge(
                    current_df,
                    how="left",
                    left_on=join_key_out,
                    right_on=join_key_in
                )

        print(f"📋 Final columns: {result_df.columns.tolist()}")
        
        # ✅ SIMPLER FIX: Just find the qualified filter column name
        if filter_column not in result_df.columns:
            # Look for the column from the first file (it will be filter_column or filter_column_firstfilename)
            first_file_columns = [col for col in result_df.columns if col.startswith(filter_column)]
            if first_file_columns:
                actual_filter_column = first_file_columns[0]  # Use the first match
                print(f"🔍 Using qualified column: {actual_filter_column}")
            else:
                raise ValueError(f"Filter column '{filter_column}' not found in data. Available: {result_df.columns.tolist()}")
        else:
            actual_filter_column = filter_column
        
        print(f"🔍 Filtering for {actual_filter_column} = {national_id}")
        target_df = result_df[result_df[actual_filter_column] == national_id]
        
        if target_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ID {national_id}")
        
        print(f"✅ Filter result: {target_df.shape} rows found")
        return target_df
        
    except Exception as e:
        print(f"❌ Error in load_and_filter_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data loading error: {str(e)}")

def call_ollama(prompt: str, task_type: str = "default", temperature: float = None) -> str:
    """Smart model selection based on task type"""
    
    # Get appropriate model for task
    model = OLLAMA_MODELS.get(task_type, OLLAMA_MODELS["default"])
    
    # Set temperature based on task type
    if temperature is None:
        temperature = 0.3 if task_type in ["sql", "dbt", "quality", "schema_mapper", "schema_harmonizer"] else 0.7
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data, timeout=120)
        response.raise_for_status()
        result_json = response.json()
        return result_json.get('response', '').strip()
    except Exception as e:
        print(f"Ollama Error (Model: {model}): {e}")
        return f"Error generating response with {model}: {str(e)}"

# --- Enhanced AI Module Functions ---
def generate_sql_code(request: SQLRequest) -> Dict[str, str]:
    """Generate SQL using SQLCoder"""
    prompt = AI_PROMPTS["sql_generation"].format(
        prompt=request.prompt,
        schema_context=request.schema_context,
        database_type=request.database_type
    )
    sql_code = call_ollama(prompt, task_type="sql", temperature=0.3)
    return {"code": sql_code, "model": OLLAMA_MODELS["sql"]}

def generate_dbt_tests(request: DBTRequest) -> Dict[str, str]:
    """Generate dbt tests using SQLCoder"""
    prompt = AI_PROMPTS["dbt_tests"].format(
        table_name=request.table_name,
        columns=", ".join(request.columns),
        requirements=request.requirements
    )
    dbt_yaml = call_ollama(prompt, task_type="dbt", temperature=0.3)
    return {"code": dbt_yaml, "model": OLLAMA_MODELS["dbt"]}

def generate_pipeline_code(request: PipelineRequest) -> Dict[str, str]:
    """Generate pipeline code using CodeLlama"""
    prompt = AI_PROMPTS["pipeline_design"].format(
        description=request.description,
        source_type=request.source_type,
        target_type=request.target_type,
        tools=", ".join(request.tools)
    )
    pipeline_code = call_ollama(prompt, task_type="pipeline", temperature=0.5)
    return {"code": pipeline_code, "model": OLLAMA_MODELS["pipeline"]}

def generate_quality_checks(request: QualityRequest) -> Dict[str, str]:
    """Generate data quality checks using SQLCoder"""
    prompt = AI_PROMPTS["data_quality"].format(
        dataset=request.dataset,
        checks=request.checks,
        framework=request.framework
    )
    quality_code = call_ollama(prompt, task_type="quality", temperature=0.3)
    return {"code": quality_code, "model": OLLAMA_MODELS["quality"]}

def generate_s3_operations(request: S3Request) -> Dict[str, str]:
    """Generate S3 operation code using CodeLlama"""
    prompt = AI_PROMPTS["s3_operations"].format(
        operation=request.operation,
        bucket=request.bucket,
        path=request.path,
        file_format=request.file_format
    )
    s3_code = call_ollama(prompt, task_type="s3", temperature=0.5)
    return {"code": s3_code, "model": OLLAMA_MODELS["s3"]}

def generate_schema_mapping(request: SchemaMapperRequest) -> Dict[str, str]:
    """Generate schema mapping using SQLCoder"""
    prompt = AI_PROMPTS["schema_mapper"].format(
        source_system=request.source_system,
        target_system=request.target_system,
        source_schema=json.dumps(request.source_schema, indent=2),
        target_schema=json.dumps(request.target_schema, indent=2),
        mapping_rules=request.mapping_rules
    )
    mapping_plan = call_ollama(prompt, task_type="schema_mapper", temperature=0.4)
    return {"plan": mapping_plan, "model": OLLAMA_MODELS["schema_mapper"]}

def generate_schema_harmonization(request: SchemaHarmonizerRequest) -> Dict[str, str]:
    """Generate schema harmonization using SQLCoder"""
    
    # Save the configuration if output path provided
    if request.output_config_path:
        output_path = Path(request.output_config_path)
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(request.config, f, indent=2)
    
    prompt = AI_PROMPTS["schema_harmonizer"].format(
        target_system=request.target_system,
        source_config=json.dumps(request.config, indent=2),
        mapping_requirements=request.mapping_requirements
    )
    
    harmonization_plan = call_ollama(prompt, task_type="schema_harmonizer", temperature=0.4)
    return {"plan": harmonization_plan, "model": OLLAMA_MODELS["schema_harmonizer"]}

def generate_chat_response(request: ChatRequest) -> Dict[str, str]:
    """Generate chat response using Gemma3"""
    prompt = f"""You are an expert data engineering assistant. Help with this question:

QUESTION: {request.message}

CONTEXT: {request.context}

Provide helpful, practical advice for data engineering tasks. Be specific and provide examples when possible."""
    
    response = call_ollama(prompt, task_type="chat", temperature=0.7)
    return {"response": response, "model": OLLAMA_MODELS["chat"]}

def load_authentication_prompts():
    """Load authentication prompts from JSON file or return defaults"""
    try:
        prompts_path = Path("authentication_prompts.json")
        if prompts_path.exists():
            print("📁 Loading prompts from authentication_prompts.json")
            with open(prompts_path, 'r') as f:
                prompts_data = json.load(f)
            # Extract just the prompt strings for backward compatibility
            prompts = {key: data["prompt"] for key, data in prompts_data.items()}
            print(f"✅ Loaded {len(prompts)} prompts from file")
            return prompts
        else:
            print("📁 Using default authentication prompts")
            return AUTHENTICATION_PROMPTS
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return AUTHENTICATION_PROMPTS

"""
def generate_authentication_questions(data_str: str, target_id: int, prompt_type: str) -> List[QAPair]:
    try:
        print(f"🔍 Generating {prompt_type} questions for ID: {target_id}")
        
        # Load prompts dynamically
        prompts = load_authentication_prompts()
        
        if prompt_type not in prompts:
            error_msg = f"Unknown prompt type: {prompt_type}. Available types: {list(prompts.keys())}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        prompt_template = prompts[prompt_type]
        print(f"✅ Loaded prompt template for {prompt_type}")
        
        # Format the prompt with actual values
        prompt = prompt_template.format(target_id=target_id, data_str=data_str)
        print(f"✅ Formatted prompt, length: {len(prompt)}")
        
        # Call Ollama with appropriate temperature (lower for more factual responses)
        temperature = 0.1  # Lower temperature for more consistent, factual responses
        print("🔄 Calling Ollama API...")
        result = call_ollama(prompt, task_type="chat", temperature=temperature)
        print(f"✅ Ollama response received, length: {len(result)}")
        print(f"📝 Raw response preview: {result[:200]}...")
        
        # Try to extract JSON from the response
        json_text = extract_json_from_response(result)
        
        if json_text:
            try:
                parsed_response = json.loads(json_text)
                print("✅ Successfully parsed JSON response")
                
                # Handle both single object and array responses
                if isinstance(parsed_response, dict):
                    parsed_response = [parsed_response]
                
                qa_pairs = []
                for item in parsed_response:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        question = item["question"]
                        answer = item["answer"]
                        
                        # ✅ Minimal cleaning - only fix obviously generic questions
                        if any(generic in question.lower() for generic in [
                            "knowledge question", "behavioral question", 
                            "multi-factor question", "security question",
                            "question 1", "question 2", "question 3"
                        ]):
                            # Create a better question based on the answer content
                            if any(word in answer.lower() for word in ['payment', 'balance', 'amount', 'transaction']):
                                question = "Can you tell me about your recent account activity?"
                            elif any(word in answer.lower() for word in ['industry', 'work', 'employment']):
                                question = "What does our records show about your employment?"
                            elif any(word in answer.lower() for word in ['date', 'opened', 'created']):
                                question = "When was your account first opened?"
                            else:
                                question = "Can you verify some details from your account information?"
                        
                        qa_pairs.append(QAPair(
                            question=question,
                            answer=answer
                        ))
                
                print(f"✅ Extracted {len(qa_pairs)} valid Q&A pairs")
                
                # Ensure we have exactly 3 pairs
                if len(qa_pairs) < 3:
                    print(f"⚠️  Only got {len(qa_pairs)} pairs, adding data-driven questions")
                    fallback_questions = create_data_driven_questions(data_str, prompt_type)
                    qa_pairs = qa_pairs + fallback_questions
                
                # Log the final questions for debugging
                for i, qa in enumerate(qa_pairs[:3], 1):
                    print(f"   {i}. Q: {qa.question}")
                    print(f"      A: {qa.answer}")
                
                return qa_pairs[:3]
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed even after extraction: {e}")
                print(f"📝 Extracted JSON text: {json_text}")
        
        # If JSON extraction fails, create meaningful Q&A pairs from the data
        print("⚠️  Using data-driven fallback Q&A generation")
        return create_data_driven_questions(data_str, prompt_type)
            
    except Exception as e:
        print(f"❌ Error in generate_authentication_questions: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_data_driven_questions(data_str, prompt_type)


   
"""

def generate_authentication_questions(data_str: str, target_id: int, prompt_type: str) -> tuple[List[QAPair], bool, Optional[QueryLog]]:
    """Generate authentication questions using Ollama - returns (questions, llm_success, query_log)"""
    llm_success = True
    query_log = None
    
    try:
        print(f"🔍 Generating {prompt_type} questions for ID: {target_id}")
        
        # Load prompts dynamically
        prompts = load_authentication_prompts()
        
        if prompt_type not in prompts:
            error_msg = f"Unknown prompt type: {prompt_type}. Available types: {list(prompts.keys())}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        prompt_template = prompts[prompt_type]
        print(f"✅ Loaded prompt template for {prompt_type}")
        
        # Format the prompt with actual values
        full_prompt = prompt_template.format(target_id=target_id, data_str=data_str)
        print(f"✅ Formatted prompt, length: {len(full_prompt)}")
        
        # Call Ollama with appropriate temperature
        temperature = 0.1
        model = OLLAMA_MODELS["chat"]
        print("🔄 Calling Ollama API...")
        result = call_ollama(full_prompt, task_type="chat", temperature=temperature)
        print(f"✅ Ollama response received, length: {len(result)}")
        
        # Create query log
        query_log = QueryLog(
            prompt_type=prompt_type,
            target_id=target_id,
            prompt_used=full_prompt[:1000] + "..." if len(full_prompt) > 1000 else full_prompt,  # Truncate if too long
            llm_response=result[:1000] + "..." if len(result) > 1000 else result,  # Truncate if too long
            llm_model=model,
            success=True,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"📝 Raw response preview: {result[:200]}...")
        
        # Try to extract JSON from the response
        json_text = extract_json_from_response(result)
        
        if json_text:
            try:
                # Validate JSON structure before parsing
                if not all(key in json_text for key in ['"question"', '"answer"']):
                    raise json.JSONDecodeError("Missing required fields", json_text, 0)
                
                parsed_response = json.loads(json_text)
                print("✅ Successfully parsed JSON response")
                
                # Handle both single object and array responses
                if isinstance(parsed_response, dict):
                    parsed_response = [parsed_response]
                
                qa_pairs = []
                valid_count = 0
                
                for item in parsed_response:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        question = item["question"]
                        answer = item["answer"]
                        
                        # Validate that both question and answer have content
                        if question.strip() and answer.strip():
                            # Minimal cleaning for obviously generic questions
                            if any(generic in question.lower() for generic in [
                                "knowledge question", "behavioral question", 
                                "multi-factor question", "security question",
                                "question 1", "question 2", "question 3"
                            ]):
                                # Create better question based on the answer content
                                if any(word in answer.lower() for word in ['payment', 'balance', 'amount', 'transaction']):
                                    question = "Can you tell me about your recent account activity?"
                                elif any(word in answer.lower() for word in ['industry', 'work', 'employment']):
                                    question = "What does our records show about your employment?"
                                elif any(word in answer.lower() for word in ['date', 'opened', 'created']):
                                    question = "When was your account first opened?"
                                else:
                                    question = "Can you verify some details from your account information?"
                            
                            qa_pairs.append(QAPair(
                                question=question,
                                answer=answer
                            ))
                            valid_count += 1
                
                print(f"✅ Extracted {valid_count} valid Q&A pairs from LLM")
                
                # If we got at least 2 valid pairs from LLM, use them
                if valid_count >= 2:
                    # Ensure we have exactly 3 pairs
                    if len(qa_pairs) < 3:
                        print(f"⚠️  LLM generated {len(qa_pairs)} pairs, adding data-driven questions")
                        fallback_questions = create_data_driven_questions(data_str, prompt_type)
                        qa_pairs = qa_pairs + fallback_questions
                        llm_success = False  # Mark as partial failure
                    
                    # Log the final questions for debugging
                    for i, qa in enumerate(qa_pairs[:3], 1):
                        print(f"   {i}. Q: {qa.question}")
                        print(f"      A: {qa.answer}")
                    
                    return qa_pairs[:3], llm_success, query_log
                else:
                    print(f"❌ LLM generated only {valid_count} valid pairs, using fallback")
                    llm_success = False
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"📝 Extracted JSON text: {json_text}")
                llm_success = False
                # Update query log to reflect failure
                if query_log:
                    query_log.success = False
        else:
            print("❌ No valid JSON found in LLM response")
            llm_success = False
            # Update query log to reflect failure
            if query_log:
                query_log.success = False
        
        # If we reach here, LLM generation failed
        print("⚠️  Using data-driven fallback Q&A generation")
        fallback_questions = create_data_driven_questions(data_str, prompt_type)
        
        return fallback_questions[:3], llm_success, query_log
            
    except Exception as e:
        print(f"❌ Error in generate_authentication_questions: {str(e)}")
        import traceback
        traceback.print_exc()
        llm_success = False
        
        # Create error query log
        query_log = QueryLog(
            prompt_type=prompt_type,
            target_id=target_id,
            prompt_used=full_prompt[:1000] + "..." if 'full_prompt' in locals() and len(full_prompt) > 1000 else full_prompt if 'full_prompt' in locals() else "Prompt not generated",
            llm_response=f"Error: {str(e)}",
            llm_model=OLLAMA_MODELS["chat"],
            success=False,
            timestamp=datetime.now().isoformat()
        )
        
        return create_data_driven_questions(data_str, prompt_type), llm_success, query_log

def create_data_driven_questions(data_str: str, prompt_type: str) -> List[QAPair]:
    """Create questions based on actual data content when LLM fails"""
    try:
        # Parse the CSV data to extract meaningful information
        lines = data_str.strip().split('\n')
        if len(lines) < 2:
            return create_error_fallback_questions()
        
        headers = [h.strip() for h in lines[0].split(',')]
        data_row = [v.strip() for v in lines[1].split(',')]  # First data row
        
        # Create a mapping of header to value
        data_dict = dict(zip(headers, data_row))
        
        questions = []
        
        if prompt_type == "knowledge":
            # Create knowledge questions based on available data fields
            if 'lastpaymentdate' in data_dict and data_dict['lastpaymentdate'] and data_dict['lastpaymentdate'].lower() not in ['', 'null', 'nan']:
                questions.append(QAPair(
                    question="When was your last payment made according to our records?",
                    answer=f"Your last payment was made on {data_dict['lastpaymentdate']}"
                ))
            
            if 'currentbalanceamt' in data_dict and data_dict['currentbalanceamt'] and data_dict['currentbalanceamt'].lower() not in ['', 'null', 'nan']:
                questions.append(QAPair(
                    question="What's the current balance showing on your account?",
                    answer=f"Your current balance is {data_dict['currentbalanceamt']}"
                ))
            
            if 'industry' in data_dict and data_dict['industry'] and data_dict['industry'].lower() not in ['', 'null', 'nan']:
                questions.append(QAPair(
                    question="Which industry sector are you employed in based on our records?",
                    answer=f"You work in the {data_dict['industry']} industry"
                ))
            
            if 'subscribername' in data_dict and data_dict['subscribername'] and data_dict['subscribername'].lower() not in ['', 'null', 'nan']:
                questions.append(QAPair(
                    question="Which company or service provider is your account registered with?",
                    answer=f"Your account is with {data_dict['subscribername']}"
                ))
            
            # Fill remaining slots if needed
            while len(questions) < 3:
                if 'accountopeneddate' in data_dict and data_dict['accountopeneddate'] and data_dict['accountopeneddate'].lower() not in ['', 'null', 'nan']:
                    questions.append(QAPair(
                        question="When did you first open this account with us?",
                        answer=f"Your account was opened on {data_dict['accountopeneddate']}"
                    ))
                elif 'status1' in data_dict and data_dict['status1'] and data_dict['status1'].lower() not in ['', 'null', 'nan']:
                    questions.append(QAPair(
                        question="What's the current status of your account?",
                        answer=f"Your account status is {data_dict['status1']}"
                    ))
                else:
                    questions.append(QAPair(
                        question="Can you verify your account details with us?",
                        answer="Based on your account records, specific verification information is available"
                    ))
        
        elif prompt_type == "behavioral":
            # Create behavioral questions
            if 'lastpaymentdate' in data_dict and data_dict['lastpaymentdate']:
                questions.append(QAPair(
                    question="Based on your payment history, when do you typically make your payments?",
                    answer=f"Your payment patterns show activity around {data_dict['lastpaymentdate']}"
                ))
            
            if any(field in data_dict for field in ['currentbalanceamt', 'openingbalanceamt', 'balanceoverdueamt']):
                questions.append(QAPair(
                    question="How would you describe your typical account balance management?",
                    answer="Your account shows specific balance patterns that help verify your identity"
                ))
            
            # Fill remaining slots
            while len(questions) < 3:
                questions.append(QAPair(
                    question="Can you describe your usual transaction patterns?",
                    answer="Your transaction history shows identifiable behavioral patterns"
                ))
        
        else:
            # For other prompt types, use generic but appropriate questions
            while len(questions) < 3:
                questions.append(QAPair(
                    question="Can you verify your identity with some account details?",
                    answer="Your account information contains specific verification data"
                ))
        
        return questions[:3]
            
    except Exception as e:
        print(f"❌ Error creating data-driven questions: {e}")
        return create_error_fallback_questions()




def extract_json_from_response(text: str) -> str:
    """Extract JSON from LLM response that might contain explanations"""
    import re
    
    # Try to find JSON array
    json_array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if json_array_match:
        return json_array_match.group(0)
    
    # Try to find JSON object
    json_object_match = re.search(r'\{\s*".*?"\s*:\s*".*?"\s*\}', text, re.DOTALL)
    if json_object_match:
        return f"[{json_object_match.group(0)}]"
    
    # Try to find code blocks with JSON
    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)
    
    return ""

def create_fallback_questions(data_str: str, existing_count: int) -> List[QAPair]:
    """Create context-aware fallback questions based on the actual data"""
    # Try to extract some context from the data
    lines = data_str.split('\n')
    if len(lines) > 1:
        headers = lines[0].split(',')
        sample_data = lines[1] if len(lines) > 1 else ""
        
        fallbacks = [
            QAPair(
                question=f"What patterns do you typically show in your {headers[0] if headers else 'account'} activity?",
                answer="Based on your data, specific behavioral patterns are observed."
            ),
            QAPair(
                question="How would you describe your typical transaction behavior?",
                answer="Your transaction history reveals consistent behavioral characteristics."
            ),
            QAPair(
                question="What unique habits are reflected in your account activity?",
                answer="Your account data shows distinctive behavioral markers."
            )
        ]
    else:
        fallbacks = [
            QAPair(question="Behavioral pattern question", answer="Pattern analysis based on your data"),
            QAPair(question="Transaction habit question", answer="Habit analysis from your activity"),
            QAPair(question="Account behavior question", answer="Behavioral insights from your profile")
        ]
    
    return fallbacks[existing_count:]

def create_enhanced_fallback_questions(llm_response: str, data_str: str) -> List[QAPair]:
    """Create better fallback questions by extracting meaningful content from LLM response"""
    # Try to extract questions and answers from the text response
    import re
    
    questions = []
    answers = []
    
    # Look for question-like patterns
    question_patterns = [
        r'\"question\"\s*:\s*\"([^\"]+)\"',
        r'Question\s*:\s*([^\n]+)',
        r'\?\s*([^\n]+)'
    ]
    
    for pattern in question_patterns:
        found = re.findall(pattern, llm_response, re.IGNORECASE)
        questions.extend(found)
    
    # Look for answer-like patterns  
    answer_patterns = [
        r'\"answer\"\s*:\s*\"([^\"]+)\"',
        r'Answer\s*:\s*([^\n]+)',
        r'Expected[^\n]+:\s*([^\n]+)'
    ]
    
    for pattern in answer_patterns:
        found = re.findall(pattern, llm_response, re.IGNORECASE)
        answers.extend(found)
    
    qa_pairs = []
    for i in range(min(3, max(len(questions), len(answers)))):
        question = questions[i] if i < len(questions) else f"Behavioral verification question {i+1}"
        answer = answers[i] if i < len(answers) else f"Verification based on your unique patterns {i+1}"
        
        # Clean up the text
        question = question.strip().strip('"').strip()
        answer = answer.strip().strip('"').strip()
        
        qa_pairs.append(QAPair(question=question, answer=answer))
    
    # Fill remaining slots if needed
    while len(qa_pairs) < 3:
        qa_pairs.append(QAPair(
            question=f"Identity verification question {len(qa_pairs)+1}",
            answer="Verification based on your behavioral data"
        ))
    
    return qa_pairs[:3]

def create_error_fallback_questions() -> List[QAPair]:
    """Create fallback questions for error scenarios"""
    return [
        QAPair(
            question="What behavioral patterns are typical for your account?",
            answer="Your account shows consistent behavioral markers for verification"
        ),
        QAPair(
            question="How would you characterize your transaction habits?", 
            answer="Your transaction history reveals identifiable behavioral patterns"
        ),
        QAPair(
            question="What unique behaviors identify your account activity?",
            answer="Your account activity demonstrates distinctive behavioral characteristics"
        )
    ]
# --- API Endpoints (Enhanced) ---
@app.get("/")
async def root():
    return {
        "message": "DataGenie AI Co-Pilot Service", 
        "status": "running", 
        "version": "2.0.0",
        "available_models": OLLAMA_MODELS
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with model availability"""
    model_status = {}
    for task, model in OLLAMA_MODELS.items():
        try:
            response = requests.post(OLLAMA_API_URL, json={
                "model": model,
                "prompt": "test",
                "stream": False
            }, timeout=5)
            model_status[model] = "available" if response.status_code == 200 else "unavailable"
        except:
            model_status[model] = "unavailable"
    
    return {
        "status": "healthy",
        "service": "DataGenie AI Co-Pilot",
        "ollama_url": OLLAMA_API_URL,
        "modules": ["sql", "dbt", "pipeline", "schema_mapper", "schema_harmonizer", "quality", "s3", "chat"],
        "model_status": model_status
    }

# --- AI Module Endpoints ---
@app.post("/ai/sql/generate")
async def ai_generate_sql(request: SQLRequest):
    """Generate SQL from natural language using SQLCoder"""
    try:
        result = generate_sql_code(request)
        return {
            "success": True, 
            "sql": result["code"], 
            "model": result["model"],
            "task_type": "sql"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/dbt/generate-tests")
async def ai_generate_dbt_tests(request: DBTRequest):
    """Generate dbt tests using SQLCoder"""
    try:
        result = generate_dbt_tests(request)
        return {
            "success": True, 
            "yaml": result["code"], 
            "model": result["model"],
            "task_type": "dbt"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/pipeline/generate")
async def ai_generate_pipeline(request: PipelineRequest):
    """Generate data pipeline using CodeLlama"""
    try:
        result = generate_pipeline_code(request)
        return {
            "success": True, 
            "code": result["code"], 
            "model": result["model"],
            "task_type": "pipeline"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/schema/map")
async def ai_map_schemas(request: SchemaMapperRequest):
    """Map schemas between source and target systems"""
    try:
        result = generate_schema_mapping(request)
        return {
            "success": True, 
            "mapping_plan": result["plan"], 
            "model": result["model"],
            "task_type": "schema_mapper"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/schema/harmonize")
async def ai_harmonize_schemas(request: SchemaHarmonizerRequest):
    """Harmonize schemas from file configuration"""
    try:
        result = generate_schema_harmonization(request)
        return {
            "success": True, 
            "harmonization_plan": result["plan"], 
            "model": result["model"],
            "task_type": "schema_harmonizer",
            "config_saved": bool(request.output_config_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/quality/generate-checks")
async def ai_generate_quality_checks(request: QualityRequest):
    """Generate data quality checks using SQLCoder"""
    try:
        result = generate_quality_checks(request)
        return {
            "success": True, 
            "checks": result["code"], 
            "model": result["model"],
            "task_type": "quality"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/s3/generate-code")
async def ai_generate_s3_code(request: S3Request):
    """Generate S3 operation code using CodeLlama"""
    try:
        result = generate_s3_operations(request)
        return {
            "success": True, 
            "code": result["code"], 
            "model": result["model"],
            "task_type": "s3"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/chat")
async def ai_chat(request: ChatRequest):
    """AI chat assistant using Gemma3"""
    try:
        result = generate_chat_response(request)
        return {
            "success": True, 
            "response": result["response"], 
            "model": result["model"],
            "task_type": "chat"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# --- Prompt Management Endpoints ---
@app.get("/admin/prompts", response_class=HTMLResponse)
async def get_prompts_management():
    """Serve the prompts management UI"""
    return get_prompts_management_ui()

@app.get("/api/prompts")
async def get_all_prompts():
    """Get all authentication prompts"""
    try:
        prompts_path = Path("authentication_prompts.json")
        if not prompts_path.exists():
            # Create default prompts file if it doesn't exist
            default_prompts = {
                "behavioral": {
                    "name": "Behavioral Authentication",
                    "description": "Generate behavioral patterns-based verification questions",
                    "prompt": AUTHENTICATION_PROMPTS["behavioral"],
                    "variables": ["target_id", "data_str"]
                },
                "knowledge": {
                    "name": "Knowledge-based Authentication", 
                    "description": "Generate knowledge-based verification questions",
                    "prompt": AUTHENTICATION_PROMPTS["knowledge"],
                    "variables": ["target_id", "data_str"]
                },
                "multifactor": {
                    "name": "Multi-factor Authentication",
                    "description": "Generate multi-factor verification questions", 
                    "prompt": AUTHENTICATION_PROMPTS["multifactor"],
                    "variables": ["target_id", "data_str"]
                },
                "security": {
                    "name": "Security-focused Authentication",
                    "description": "Generate security-focused verification questions",
                    "prompt": AUTHENTICATION_PROMPTS["security"],
                    "variables": ["target_id", "data_str"]
                }
            }
            with open(prompts_path, 'w') as f:
                json.dump(default_prompts, f, indent=2)
            return {"prompts": default_prompts}
        
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompts: {str(e)}")

@app.post("/api/prompts/{prompt_type}")
async def update_prompt(prompt_type: str, prompt_data: dict):
    """Update a specific prompt"""
    try:
        prompts_path = Path("authentication_prompts.json")
        
        # Load existing prompts
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                all_prompts = json.load(f)
        else:
            all_prompts = {}
        
        # Update the specific prompt
        if prompt_type in all_prompts:
            # Only allow updating certain fields (not variables)
            allowed_fields = ["name", "description", "prompt"]
            for field in allowed_fields:
                if field in prompt_data:
                    all_prompts[prompt_type][field] = prompt_data[field]
            
            # Save updated prompts
            with open(prompts_path, 'w') as f:
                json.dump(all_prompts, f, indent=2)
            
            return {"status": "success", "message": f"Prompt '{prompt_type}' updated successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Prompt type '{prompt_type}' not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prompt: {str(e)}")

@app.post("/api/prompts/reset/{prompt_type}")
async def reset_prompt(prompt_type: str):
    """Reset a prompt to its default value"""
    try:
        # Check if prompt type exists in defaults
        if prompt_type not in AUTHENTICATION_PROMPTS:
            raise HTTPException(status_code=404, detail=f"Prompt type '{prompt_type}' not found")
        
        prompts_path = Path("authentication_prompts.json")
        
        # Load existing prompts or create new
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                all_prompts = json.load(f)
        else:
            all_prompts = {}
        
        # Reset to default
        all_prompts[prompt_type] = {
            "name": prompt_type.replace('_', ' ').title() + " Authentication",
            "description": f"Default {prompt_type} authentication prompt",
            "prompt": AUTHENTICATION_PROMPTS[prompt_type],
            "variables": ["target_id", "data_str"]
        }
        
        # Save
        with open(prompts_path, 'w') as f:
            json.dump(all_prompts, f, indent=2)
        
        return {"status": "success", "message": f"Prompt '{prompt_type}' reset to default"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting prompt: {str(e)}")
    
# --- Authentication Endpoints with Query Logging ---
@app.get("/auth/behavioral/personal/{national_id}", response_model=ReportResponse)
async def behavioral_auth_personal(national_id: int):
    try:
        print(f"🔍 Starting behavioral auth for ID: {national_id}")
        target_df = load_and_filter_data(national_id, "config_personal.json")
        print(f"✅ Data loaded successfully, shape: {target_df.shape}")
        
        data_str = target_df.to_csv(index=False, header=True)
        print(f"✅ Data converted to CSV, length: {len(data_str)}")
        
        qa_pairs, llm_success, query_log = generate_authentication_questions(data_str, national_id, "behavioral")
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        return ReportResponse(
            report_id=national_id, 
            llm_model=OLLAMA_MODELS["chat"], 
            results=qa_pairs,
            llm_generation_success=llm_success,
            query_log=query_log
        )
    except Exception as e:
        print(f"❌ Error in behavioral_auth_personal: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/knowledge/personal/{national_id}", response_model=ReportResponse)
async def knowledge_auth_personal(national_id: int):
    try:
        print(f"🔍 Starting knowledge auth for ID: {national_id}")
        target_df = load_and_filter_data(national_id, "config_personal.json")
        print(f"✅ Data loaded successfully, shape: {target_df.shape}")
        
        data_str = target_df.to_csv(index=False, header=True)
        print(f"✅ Data converted to CSV, length: {len(data_str)}")
        
        qa_pairs, llm_success, query_log = generate_authentication_questions(data_str, national_id, "knowledge")
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        return ReportResponse(
            report_id=national_id, 
            llm_model=OLLAMA_MODELS["chat"], 
            results=qa_pairs,
            llm_generation_success=llm_success,
            query_log=query_log
        )
    except Exception as e:
        print(f"❌ Error in knowledge_auth_personal: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/multifactor/personal/{national_id}", response_model=ReportResponse)
async def multifactor_auth_personal(national_id: int):
    try:
        print(f"🔍 Starting multifactor auth for ID: {national_id}")
        target_df = load_and_filter_data(national_id, "config_personal.json")
        print(f"✅ Data loaded successfully, shape: {target_df.shape}")
        
        data_str = target_df.to_csv(index=False, header=True)
        print(f"✅ Data converted to CSV, length: {len(data_str)}")
        
        qa_pairs, llm_success, query_log = generate_authentication_questions(data_str, national_id, "multifactor")
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        return ReportResponse(
            report_id=national_id, 
            llm_model=OLLAMA_MODELS["chat"], 
            results=qa_pairs,
            llm_generation_success=llm_success,
            query_log=query_log
        )
    except Exception as e:
        print(f"❌ Error in multifactor_auth_personal: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/security/personal/{national_id}", response_model=ReportResponse)
async def security_auth_personal(national_id: int):
    try:
        print(f"🔍 Starting security auth for ID: {national_id}")
        target_df = load_and_filter_data(national_id, "config_personal.json")
        print(f"✅ Data loaded successfully, shape: {target_df.shape}")
        
        data_str = target_df.to_csv(index=False, header=True)
        print(f"✅ Data converted to CSV, length: {len(data_str)}")
        
        qa_pairs, llm_success, query_log = generate_authentication_questions(data_str, national_id, "security")
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        return ReportResponse(
            report_id=national_id, 
            llm_model=OLLAMA_MODELS["chat"], 
            results=qa_pairs,
            llm_generation_success=llm_success,
            query_log=query_log
        )
    except Exception as e:
        print(f"❌ Error in security_auth_personal: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Admin Endpoints (Keep original functionality) ---
@app.get("/admin/files", response_model=List[FileInfo])
async def list_csv_files(base_path: str):
    try:
        base_path = Path(base_path)
        if not base_path.exists():
            raise HTTPException(status_code=400, detail="Directory not found")
        
        csv_files = glob.glob(str(base_path / "*.csv"))
        file_info_list = []
        
        for file_path in csv_files:
            file_path = Path(file_path)
            try:
                df = pd.read_csv(file_path)
                columns = df.columns.tolist()
                file_info_list.append(FileInfo(
                    name=file_path.stem,
                    path=str(file_path.relative_to(base_path)),
                    columns=columns
                ))
            except Exception as e:
                continue  # Skip invalid CSV files
                
        return file_info_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/save-config")
async def save_config(config: ConfigInput):
    try:
        output_path = Path(config.output_config_path)
        output_dir = output_path.parent
        
        # Create output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            
        if output_path.exists() and not os.access(output_path, os.W_OK):
            raise HTTPException(status_code=403, detail=f"No write permission for {output_path}")
            
        if not config.join_order:
            raise HTTPException(status_code=400, detail="Join order cannot be empty")
            
        # Validate file names in join order
        file_names = {file["name"] for file in config.files}
        for name in config.join_order:
            if name not in file_names:
                raise HTTPException(status_code=400, detail=f"Join order contains unknown file name: {name}")
                
        # Validate files exist and columns are correct
        base_path = Path(config.base_path)
        for file_info in config.files:
            file_path = base_path / file_info["path"]
            if not file_path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
                
            # Try to read the file to validate columns
            try:
                df = pd.read_csv(file_path)
                columns = df.columns.tolist()
                
                if file_info.get("primary_key") and file_info["primary_key"] not in columns:
                    raise HTTPException(status_code=400, detail=f"Primary key '{file_info['primary_key']}' not found in {file_info['name']}")
                    
                if file_info.get("join_key_in") and file_info["join_key_in"] not in columns:
                    raise HTTPException(status_code=400, detail=f"Join key in '{file_info['join_key_in']}' not found in {file_info['name']}")
                    
                if file_info.get("join_key_out") and file_info["join_key_out"] not in columns:
                    raise HTTPException(status_code=400, detail=f"Join key out '{file_info['join_key_out']}' not found in {file_info['name']}")
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading {file_info['name']}: {str(e)}")
                
        # Validate join configuration
        for i in range(1, len(config.join_order)):
            prev_file = next(f for f in config.files if f["name"] == config.join_order[i-1])
            current_file = next(f for f in config.files if f["name"] == config.join_order[i])
            
            if i < len(config.join_order) - 1 and not prev_file.get("join_key_out"):
                raise HTTPException(status_code=400, detail=f"Missing join_key_out for {prev_file['name']}")
                
            if not current_file.get("join_key_in"):
                raise HTTPException(status_code=400, detail=f"Missing join_key_in for {current_file['name']}")
                
        if not config.files[0].get("primary_key"):
            raise HTTPException(status_code=400, detail="Primary key required for the first file")
            
        # Save the configuration
        config_dict = {
            "path": config.base_path,
            "files": config.files,
            "join_order": config.join_order
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        return {"message": f"Configuration saved to {output_path}", "status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")
    
# --- Enhanced Main UI ---
@app.get("/ui", response_class=HTMLResponse)
async def main_ui():
    """Main DataGenie AI Co-Pilot Interface"""
    return get_ai_copilot_ui()

@app.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DataGenie Config Creator</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .draggable { cursor: move; padding: 8px; background: #e5e7eb; border-radius: 4px; margin-bottom: 4px; }
            .dropzone { min-height: 100px; border: 2px dashed #d1d5db; padding: 8px; border-radius: 4px; }
            .dragover { border-color: #3b82f6; background: #eff6ff; }
            .conflict-warning { color: #dc2626; font-size: 0.875rem; margin-top: 4px; }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen p-6">
        <div class="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-md">
            <h1 class="text-3xl font-bold text-gray-800 mb-6">DataGenie Config Creator</h1>
            
            <div class="mb-4">
                <label class="block text-gray-700 font-semibold mb-2">Base Path:</label>
                <p class="text-sm text-gray-600 mb-2">The directory containing your CSV files (e.g., C:/data/personal).</p>
                <input id="basePath" type="text" placeholder="e.g., C:/data/personal or /home/user/data" 
                       class="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                <button onclick="listFiles()" 
                        class="mt-2 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                    List CSV Files
                </button>
            </div>
            
            <div id="fileSelection" class="mb-4 hidden">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Select Files to Configure</h3>
                <p class="text-sm text-gray-600 mb-2">Choose which CSV files to include in the configuration.</p>
                <div id="fileCheckboxes" class="space-y-2"></div>
                <button onclick="configureSelectedFiles()" 
                        class="mt-2 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                    Configure Selected Files
                </button>
            </div>
            
            <div id="configSection" class="hidden">
                <div id="fileList" class="space-y-4"></div>
                
                <div id="joinOrder" class="mt-6">
                    <h3 class="text-lg font-semibold text-gray-700">Join Order</h3>
                    <p class="text-sm text-gray-600 mb-2">Drag and drop files to set the join order (e.g., accounts → orders → payments).</p>
                    <div id="joinOrderDropzone" class="dropzone" ondragover="event.preventDefault(); this.classList.add('dragover');" 
                         ondragleave="this.classList.remove('dragover');" ondrop="dropFile(event)">
                    </div>
                </div>
                
                <div id="conflictWarning" class="conflict-warning hidden"></div>
                
                <div class="mt-4">
                    <label class="block text-gray-700 font-semibold mb-2">Filter Column:</label>
                    <p class="text-sm text-gray-600 mb-2">Select a column from the first file to filter data (e.g., PersonNationalID).</p>
                    <select id="filterColumn" 
                            class="w-full p-2 border rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="">Select a column</option>
                    </select>
                </div>
                
                <div class="mt-4">
                    <label class="block text-gray-700 font-semibold mb-2">Output Config Path:</label>
                    <p class="text-sm text-gray-600 mb-2">Where to save the config file (e.g., config_personal.json).</p>
                    <input id="outputConfigPath" type="text" placeholder="e.g., config_personal.json" 
                           class="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <button onclick="saveConfig()" 
                            class="mt-2 bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600">
                        Save Config
                    </button>
                </div>
            </div>
        </div>
        
        <script>
            let allFiles = [];
            let joinOrder = [];
            
            async function listFiles() {
                const basePath = document.getElementById('basePath').value;
                if (!basePath) {
                    alert('Please enter a base path.');
                    return;
                }
                try {
                    const response = await fetch(`/admin/files?base_path=${encodeURIComponent(basePath)}`);
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(errorText);
                    }
                    allFiles = await response.json();
                    const fileCheckboxes = document.getElementById('fileCheckboxes');
                    fileCheckboxes.innerHTML = '';
                    
                    if (allFiles.length === 0) {
                        fileCheckboxes.innerHTML = '<p class="text-gray-600">No CSV files found in the specified directory.</p>';
                    } else {
                        allFiles.forEach((file, index) => {
                            const div = document.createElement('div');
                            div.innerHTML = `
                                <label class="flex items-center space-x-2">
                                    <input type="checkbox" id="file_${index}" value="${file.name}" class="h-4 w-4">
                                    <span>${file.name}.csv (Columns: ${file.columns.join(', ')})</span>
                                </label>
                            `;
                            fileCheckboxes.appendChild(div);
                        });
                    }
                    
                    document.getElementById('fileSelection').classList.remove('hidden');
                    document.getElementById('configSection').classList.add('hidden');
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            function configureSelectedFiles() {
                const selectedFiles = Array.from(document.querySelectorAll('#fileCheckboxes input:checked'))
                    .map(input => allFiles.find(file => file.name === input.value));
                    
                if (selectedFiles.length === 0) {
                    alert('Please select at least one file.');
                    return;
                }
                
                joinOrder = selectedFiles.map(file => file.name);
                const fileListDiv = document.getElementById('fileList');
                fileListDiv.innerHTML = '';
                const joinOrderDropzone = document.getElementById('joinOrderDropzone');
                joinOrderDropzone.innerHTML = '';
                
                const columnCounts = {};
                selectedFiles.forEach(file => {
                    file.columns.forEach(col => {
                        columnCounts[col] = (columnCounts[col] || 0) + 1;
                    });
                });
                
                const conflicts = Object.keys(columnCounts).filter(col => columnCounts[col] > 1);
                const conflictWarning = document.getElementById('conflictWarning');
                if (conflicts.length > 0) {
                    conflictWarning.innerText = `Warning: Columns ${conflicts.join(', ')} appear in multiple files and may cause conflicts.`;
                    conflictWarning.classList.remove('hidden');
                } else {
                    conflictWarning.classList.add('hidden');
                }
                
                // Populate filter column dropdown with columns from the first file in join order
                const firstFile = selectedFiles.find(file => file.name === joinOrder[0]);
                let firstFileColumns = Array.isArray(firstFile.columns) ? firstFile.columns : firstFile.columns.split(",").map(col => col.trim());
                const filterColumnSelect = document.getElementById('filterColumn');
                filterColumnSelect.innerHTML = '<option value="">Select a column</option>' +
                    firstFileColumns.map(col => `<option value="${col}">${col}</option>`).join('');
                
                selectedFiles.forEach((file, index) => {
                    let columns = Array.isArray(file.columns) ? file.columns : file.columns.split(",").map(col => col.trim());
                    const fileDiv = document.createElement('div');
                    fileDiv.className = 'bg-gray-50 p-4 rounded-md border';
                    fileDiv.innerHTML = `
                        <h3 class="text-lg font-semibold text-gray-700 mb-2">File ${index + 1}</h3>
                        <label class="block text-gray-600">Name:</label>
                        <p class="text-sm text-gray-600 mb-2">User-defined name for the file (e.g., accounts).</p>
                        <input id="name_${index}" value="${file.name}" 
                               class="w-full p-2 border rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <label class="block text-gray-600">Path:</label>
                        <p class="text-sm text-gray-600 mb-2">Relative path to the CSV file, read-only (e.g., accounts.csv).</p>
                        <input id="path_${index}" value="${file.path}" readonly 
                               class="w-full p-2 border rounded-md mb-2 bg-gray-200">
                        <label class="block text-gray-600">Primary Key (optional):</label>
                        <p class="text-sm text-gray-600 mb-2">Column identifying unique records, required for the first file (e.g., PersonNationalID).</p>
                        <select id="primaryKey_${index}" 
                                class="w-full p-2 border rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="">None</option>
                            ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                        </select>
                        <label class="block text-gray-600">Join Key In (optional):</label>
                        <p class="text-sm text-gray-600 mb-2">Column matching the previous file's Join Key Out, required for non-first files (e.g., AccountID).</p>
                        <select id="joinKeyIn_${index}" 
                                class="w-full p-2 border rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="">None</option>
                            ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                        </select>
                        <label class="block text-gray-600">Join Key Out (optional):</label>
                        <p class="text-sm text-gray-600 mb-2">Column used to join with the next file, required for non-last files (e.g., AccountID).</p>
                        <select id="joinKeyOut_${index}" 
                                class="w-full p-2 border rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="">None</option>
                            ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                        </select>
                    `;
                    fileListDiv.appendChild(fileDiv);
                    
                    const dragItem = document.createElement('div');
                    dragItem.className = 'draggable';
                    dragItem.draggable = true;
                    dragItem.id = `drag_${file.name}`;
                    dragItem.innerText = file.name;
                    dragItem.ondragstart = (ev) => ev.dataTransfer.setData('text/plain', file.name);
                    joinOrderDropzone.appendChild(dragItem);
                });
                
                document.getElementById('fileSelection').classList.add('hidden');
                document.getElementById('configSection').classList.remove('hidden');
            }
            
            function dropFile(event) {
                event.preventDefault();
                event.target.classList.remove('dragover');
                const fileName = event.dataTransfer.getData('text/plain');
                const index = joinOrder.indexOf(fileName);
                if (index !== -1) {
                    joinOrder.splice(index, 1);
                }
                const dropzone = document.getElementById('joinOrderDropzone');
                const draggedElement = document.getElementById(`drag_${fileName}`);
                dropzone.appendChild(draggedElement);
                const items = Array.from(dropzone.children).map(item => item.innerText);
                joinOrder = items;

                // Update filter column dropdown when join order changes
                const firstFile = allFiles.find(file => file.name === joinOrder[0]);
                let firstFileColumns = firstFile ? (Array.isArray(firstFile.columns) ? firstFile.columns : firstFile.columns.split(",").map(col => col.trim())) : [];
                const filterColumnSelect = document.getElementById('filterColumn');
                filterColumnSelect.innerHTML = '<option value="">Select a column</option>' +
                    firstFileColumns.map(col => `<option value="${col}">${col}</option>`).join('');
            }
            
            async function saveConfig() {
                const basePath = document.getElementById('basePath').value;
                const outputConfigPath = document.getElementById('outputConfigPath').value;
                const filterColumn = document.getElementById('filterColumn').value;
                
                if (!basePath || !outputConfigPath || !filterColumn) {
                    alert('Please enter base path, output config path, and select a filter column.');
                    return;
                }
                
                if (joinOrder.length === 0) {
                    alert('Please define the join order by dragging files.');
                    return;
                }
                
                const files = [];
                document.querySelectorAll('#fileList .bg-gray-50').forEach((_, index) => {
                    const fileInfo = {
                        name: document.getElementById(`name_${index}`).value,
                        path: document.getElementById(`path_${index}`).value
                    };
                    
                    const primaryKey = document.getElementById(`primaryKey_${index}`).value;
                    const joinKeyIn = document.getElementById(`joinKeyIn_${index}`).value;
                    const joinKeyOut = document.getElementById(`joinKeyOut_${index}`).value;
                    
                    if (primaryKey) fileInfo.primary_key = primaryKey;
                    if (joinKeyIn) fileInfo.join_key_in = joinKeyIn;
                    if (joinKeyOut) fileInfo.join_key_out = joinKeyOut;
                    
                    files.push(fileInfo);
                });
                
                // Validate configuration
                for (let i = 1; i < joinOrder.length; i++) {
                    const prevFile = files.find(f => f.name === joinOrder[i-1]);
                    const currentFile = files.find(f => f.name === joinOrder[i]);
                    
                    if (i < joinOrder.length - 1 && !prevFile.join_key_out) {
                        alert(`Please select a Join Key Out for ${prevFile.name}`);
                        return;
                    }
                    
                    if (!currentFile.join_key_in) {
                        alert(`Please select a Join Key In for ${currentFile.name}`);
                        return;
                    }
                }
                
                if (!files[0].primary_key) {
                    alert('Please select a Primary Key for the first file.');
                    return;
                }
                
                const config = { 
                    base_path: basePath, 
                    files, 
                    join_order: joinOrder, 
                    output_config_path: outputConfigPath,
                    filter_column: filterColumn
                };
                
                try {
                    const response = await fetch('/admin/save-config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(config)
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(errorText);
                    }
                    
                    const result = await response.json();
                    alert('Config saved successfully: ' + result.message);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)




def get_ai_copilot_ui():
    """Return the complete AI Co-Pilot HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DataGenie AI Co-Pilot</title>
        <style>
            :root {
                --primary: #2563eb;
                --secondary: #64748b;
                --success: #10b981;
                --warning: #f59e0b;
                --error: #ef4444;
                --background: #f8fafc;
                --card: #ffffff;
                --text: #1e293b;
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--background); color: var(--text); line-height: 1.6; }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 40px; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
            .tabs { display: flex; background: var(--card); border-radius: 10px 10px 0 0; overflow-x: auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .tab { padding: 15px 25px; background: none; border: none; cursor: pointer; font-size: 1rem; font-weight: 500; color: var(--secondary); border-bottom: 3px solid transparent; transition: all 0.3s ease; white-space: nowrap; }
            .tab:hover { color: var(--primary); background: #f1f5f9; }
            .tab.active { color: var(--primary); border-bottom-color: var(--primary); background: #f8fafc; }
            .tab-content { display: none; padding: 30px; background: var(--card); border-radius: 0 0 10px 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .tab-content.active { display: block; }
            .card { background: var(--card); padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 25px; }
            .card h3 { margin-bottom: 15px; color: var(--primary); display: flex; align-items: center; gap: 10px; }
            .form-group { margin-bottom: 20px; }
            .form-group label { display: block; margin-bottom: 8px; font-weight: 500; color: var(--text); }
            .form-control { width: 100%; padding: 12px 15px; border: 2px solid #e2e8f0; border-radius: 8px; font-size: 1rem; transition: border-color 0.3s ease; }
            .form-control:focus { outline: none; border-color: var(--primary); }
            textarea.form-control { min-height: 100px; resize: vertical; font-family: 'Courier New', monospace; }
            .btn { padding: 12px 25px; background: var(--primary); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: 500; transition: all 0.3s ease; }
            .btn:hover { background: #1d4ed8; transform: translateY(-2px); }
            .btn:disabled { background: var(--secondary); cursor: not-allowed; transform: none; }
            .code-output { background: #1a202c; color: #e2e8f0; padding: 20px; border-radius: 8px; margin-top: 20px; font-family: 'Courier New', monospace; white-space: pre-wrap; overflow-x: auto; position: relative; }
            .copy-btn { position: absolute; top: 10px; right: 10px; background: #4a5568; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }
            .copy-btn:hover { background: #2d3748; }
            .loading { display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid var(--primary); border-radius: 50%; animation: spin 1s linear infinite; margin-right: 10px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .status { padding: 10px 15px; border-radius: 5px; margin: 10px 0; }
            .status.success { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
            .status.error { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
            .message { padding: 12px 15px; margin: 10px 0; border-radius: 10px; max-width: 80%; }
            .user-message { background: #e3f2fd; margin-left: auto; text-align: right; }
            .ai-message { background: #f1f5f9; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .feature-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
            .feature-icon { font-size: 2rem; margin-bottom: 10px; }
            .model-badge { background: #6b7280; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-left: 8px; }
            .draggable { cursor: move; padding: 8px; background: #e5e7eb; border-radius: 4px; margin-bottom: 4px; }
            .dropzone { min-height: 100px; border: 2px dashed #d1d5db; padding: 8px; border-radius: 4px; }
            .dragover { border-color: #3b82f6; background: #eff6ff; }
            .conflict-warning { color: #dc2626; font-size: 0.875rem; margin-top: 4px; }
            .hidden { display: none; }
            .grid { display: grid; }
            .grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .gap-2 { gap: 0.5rem; }
            .gap-4 { gap: 1rem; }
            .gap-6 { gap: 1.5rem; }
            .space-y-2 > * + * { margin-top: 0.5rem; }
            .space-y-4 > * + * { margin-top: 1rem; }
            .mb-2 { margin-bottom: 0.5rem; }
            .mb-4 { margin-bottom: 1rem; }
            .mt-2 { margin-top: 0.5rem; }
            .mt-4 { margin-top: 1rem; }
            .mt-6 { margin-top: 1.5rem; }
            .block { display: block; }
            .flex { display: flex; }
            .items-center { align-items: center; }
            .w-full { width: 100%; }
            .h-4 { height: 1rem; }
            .w-4 { width: 1rem; }
            .p-2 { padding: 0.5rem; }
            .p-4 { padding: 1rem; }
            .px-4 { padding-left: 1rem; padding-right: 1rem; }
            .py-2 { padding-top: 0.5rem; padding-bottom: 0.5rem; }
            .text-sm { font-size: 0.875rem; }
            .text-lg { font-size: 1.125rem; }
            .text-gray-600 { color: #4b5563; }
            .text-gray-700 { color: #374151; }
            .bg-gray-50 { background-color: #f9fafb; }
            .bg-blue-500 { background-color: #3b82f6; }
            .hover\\:bg-blue-600:hover { background-color: #2563eb; }
            .border { border-width: 1px; }
            .rounded-md { border-radius: 0.375rem; }
            @media (max-width: 768px) { 
                .container { padding: 10px; } 
                .header h1 { font-size: 2rem; } 
                .tabs { flex-wrap: wrap; } 
                .tab { flex: 1; min-width: 120px; text-align: center; } 
                .message { max-width: 95%; } 
                .grid-cols-2 { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 DataGenie AI Co-Pilot</h1>
                <p>Smart AI-powered data engineering assistant</p>
                <div style="margin-top: 20px; font-size: 0.9rem; opacity: 0.8;">
                    🚀 Multi-Model AI • SQLCoder + CodeLlama + Gemma3
                    <br>
        <a href="/admin/prompts" style="color: white; text-decoration: underline; margin-top: 10px; display: inline-block;">
            🔧 Manage Authentication Prompts
        </a>
                </div>
            </div>

            <!-- Quick Features Overview -->
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🧮</div>
                    <h4>SQL Generation <span class="model-badge">SQLCoder</span></h4>
                    <p>Optimized SQL queries with specialized model</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔀</div>
                    <h4>Schema Mapper <span class="model-badge">SQLCoder</span></h4>
                    <p>Manual schema mapping between systems</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔄</div>
                    <h4>Schema Harmonizer <span class="model-badge">SQLCoder</span></h4>
                    <p>File-based schema analysis and harmonization</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <h4>AI Assistant <span class="model-badge">Gemma3</span></h4>
                    <p>Conversational help and explanations</p>
                </div>
            </div>

            <!-- Tab Navigation -->
            <div class="tabs">
                <button class="tab active" onclick="openTab('sql-tab')">🧮 SQL Generator</button>
                <button class="tab" onclick="openTab('dbt-tab')">✅ dbt Tests</button>
                <button class="tab" onclick="openTab('pipeline-tab')">⚡ Pipelines</button>
                <button class="tab" onclick="openTab('mapper-tab')">🔀 Schema Mapper</button>
                <button class="tab" onclick="openTab('harmonizer-tab')">🔄 Schema Harmonizer</button>
                <button class="tab" onclick="openTab('s3-tab')">📁 S3 & Data Lakes</button>
                <button class="tab" onclick="openTab('quality-tab')">🔍 Data Quality</button>
                <button class="tab" onclick="openTab('chat-tab')">💬 AI Assistant</button>
            </div>

            <!-- SQL Generator Tab -->
            <div id="sql-tab" class="tab-content active">
                <div class="card">
                    <h3>🚀 Generate SQL from Natural Language <span class="model-badge">SQLCoder</span></h3>
                    <div class="form-group">
                        <label for="sqlPrompt">Describe what data you need:</label>
                        <textarea id="sqlPrompt" class="form-control" placeholder="e.g., Show me monthly revenue by customer segment for the last 6 months, including average order value...">Monthly revenue by customer segment for current year</textarea>
                    </div>
                    <div class="form-group">
                        <label for="schemaContext">Schema Context (optional):</label>
                        <textarea id="schemaContext" class="form-control" placeholder="Table names, columns, relationships...">Tables: customers(customer_id, segment, signup_date), orders(order_id, customer_id, order_date, amount)</textarea>
                    </div>
                    <div class="form-group">
                        <label for="databaseType">Target Database:</label>
                        <select id="databaseType" class="form-control">
                            <option value="snowflake">Snowflake</option>
                            <option value="bigquery">BigQuery</option>
                            <option value="redshift">Redshift</option>
                            <option value="postgresql">PostgreSQL</option>
                            <option value="mysql">MySQL</option>
                        </select>
                    </div>
                    <button class="btn" onclick="generateSQL()" id="sqlBtn">Generate SQL</button>
                    <div id="sqlStatus" class="status" style="display: none;"></div>
                    <div id="sqlOutput" class="code-output" style="display: none;">
                        <button class="copy-btn" onclick="copyCode('sqlOutput')">Copy</button>
                    </div>
                </div>
            </div>

            <!-- dbt Tests Tab -->
            <div id="dbt-tab" class="tab-content">
                <div class="card">
                    <h3>🛡️ Generate dbt Data Quality Tests <span class="model-badge">SQLCoder</span></h3>
                    <div class="form-group">
                        <label for="tableName">Table Name:</label>
                        <input type="text" id="tableName" class="form-control" placeholder="e.g., dim_customers" value="dim_customers">
                    </div>
                    <div class="form-group">
                        <label for="tableColumns">Columns (comma-separated):</label>
                        <textarea id="tableColumns" class="form-control" placeholder="e.g., customer_id, email, signup_date, status">customer_id, email, first_name, last_name, signup_date, status, lifetime_value</textarea>
                    </div>
                    <div class="form-group">
                        <label for="testRequirements">Specific Test Requirements:</label>
                        <textarea id="testRequirements" class="form-control" placeholder="e.g., Check for duplicate emails, ensure status is either 'active' or 'inactive', validate signup_date is not in future">Check for duplicate emails, ensure status is either 'active' or 'inactive', validate signup_date is not in future</textarea>
                    </div>
                    <button class="btn" onclick="generateDBTTests()" id="dbtBtn">Generate dbt Tests</button>
                    <div id="dbtStatus" class="status" style="display: none;"></div>
                    <div id="dbtOutput" class="code-output" style="display: none;">
                        <button class="copy-btn" onclick="copyCode('dbtOutput')">Copy</button>
                    </div>
                </div>
            </div>

            <!-- Pipeline Generator Tab -->
            <div id="pipeline-tab" class="tab-content">
                <div class="card">
                    <h3>⚡ Generate Data Pipeline <span class="model-badge">CodeLlama</span></h3>
                    <div class="form-group">
                        <label for="pipelineDescription">Pipeline Description:</label>
                        <textarea id="pipelineDescription" class="form-control" placeholder="e.g., Daily pipeline that extracts customer data from PostgreSQL, transforms it, and loads to Snowflake">Daily pipeline that extracts order data from MySQL, cleans duplicates, calculates metrics, and loads to BigQuery</textarea>
                    </div>
                    <div class="form-group">
                        <label for="pipelineTools">Preferred Tools:</label>
                        <select id="pipelineTools" class="form-control" multiple style="height: 100px;">
                            <option value="airflow" selected>Apache Airflow</option>
                            <option value="dbt" selected>dbt</option>
                            <option value="spark">Apache Spark</option>
                            <option value="pandas">Pandas</option>
                            <option value="docker">Docker</option>
                        </select>
                    </div>
                    <button class="btn" onclick="generatePipeline()" id="pipelineBtn">Generate Pipeline</button>
                    <div id="pipelineStatus" class="status" style="display: none;"></div>
                    <div id="pipelineOutput" class="code-output" style="display: none;">
                        <button class="copy-btn" onclick="copyCode('pipelineOutput')">Copy</button>
                    </div>
                </div>
            </div>

            <!-- Schema Mapper Tab (Manual Schema Mapping) -->
            <div id="mapper-tab" class="tab-content">
                <div class="card">
                    <h3>🔀 Schema Mapper <span class="model-badge">SQLCoder</span></h3>
                    <p class="text-gray-600 mb-4">Manually map schemas between different systems</p>
                    
                    <div class="grid grid-cols-2 gap-6">
                        <!-- Source Schema -->
                        <div class="form-group">
                            <label class="block text-gray-700 font-semibold mb-2">Source System:</label>
                            <input type="text" id="sourceSystem" class="form-control" placeholder="e.g., MySQL, PostgreSQL, Salesforce" value="PostgreSQL">
                            
                            <label class="block text-gray-700 font-semibold mb-2 mt-4">Source Schema (JSON):</label>
                            <textarea id="sourceSchema" class="form-control" rows="12" placeholder='{"column_name": "data_type", "user_id": "integer", "email": "varchar(255)"}'>{
  "customer_id": "integer",
  "first_name": "varchar(100)",
  "last_name": "varchar(100)", 
  "email": "varchar(255)",
  "signup_date": "timestamp",
  "status": "varchar(20)",
  "total_orders": "integer",
  "lifetime_value": "decimal(10,2)"
}</textarea>
                        </div>

                        <!-- Target Schema -->
                        <div class="form-group">
                            <label class="block text-gray-700 font-semibold mb-2">Target System:</label>
                            <input type="text" id="targetSystem" class="form-control" placeholder="e.g., Snowflake, BigQuery, Redshift" value="Snowflake">
                            
                            <label class="block text-gray-700 font-semibold mb-2 mt-4">Target Schema (JSON):</label>
                            <textarea id="targetSchema" class="form-control" rows="12" placeholder='{"column_name": "data_type", "user_id": "number", "email": "string"}'>{
  "cust_id": "number",
  "fname": "string",
  "lname": "string",
  "email_address": "string", 
  "created_at": "timestamp_ntz",
  "account_status": "string",
  "order_count": "number",
  "total_spent": "float"
}</textarea>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="block text-gray-700 font-semibold mb-2">Mapping Rules & Requirements:</label>
                        <textarea id="mappingRules" class="form-control" placeholder="Any specific mapping rules, business logic, or transformation requirements...">- Map customer_id to cust_id
- Combine first_name and last_name into full_name
- Transform status codes: active=1, inactive=0
- Convert timestamp to Snowflake timestamp_ntz
- Handle email validation</textarea>
                    </div>

                    <button class="btn" onclick="mapSchemas()" id="mapperBtn">Map Schemas</button>
                    <div id="mapperStatus" class="status" style="display: none;"></div>
                    <div id="mapperOutput" class="code-output" style="display: none;">
                        <button class="copy-btn" onclick="copyCode('mapperOutput')">Copy</button>
                    </div>
                </div>
            </div>

            <!-- Schema Harmonizer Tab (File-based with UI) -->
            <div id="harmonizer-tab" class="tab-content">
                <div class="card">
                    <h3>🔄 Schema Harmonizer <span class="model-badge">SQLCoder</span></h3>
                    <p class="text-gray-600 mb-4">Analyze multiple data sources and generate harmonized target schemas</p>
                    
                    <!-- File Selection Section -->
                    <div class="mb-4">
                        <label class="block text-gray-700 font-semibold mb-2">Data Directory Path:</label>
                        <p class="text-sm text-gray-600 mb-2">The directory containing your CSV files to harmonize.</p>
                        <input id="harmonizerBasePath" type="text" placeholder="e.g., C:/data/sources or /home/user/data" 
                               class="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <button onclick="listFilesForHarmonizer()" 
                                class="mt-2 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                            List CSV Files
                        </button>
                    </div>
                    
                    <div id="harmonizerFileSelection" class="mb-4 hidden">
                        <h3 class="text-lg font-semibold text-gray-700 mb-2">Select Files to Harmonize</h3>
                        <p class="text-sm text-gray-600 mb-2">Choose which CSV files to include in schema analysis.</p>
                        <div id="harmonizerFileCheckboxes" class="space-y-2"></div>
                        <button onclick="configureFilesForHarmonizer()" 
                                class="mt-2 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                            Configure File Relationships
                        </button>
                    </div>
                    
                    <!-- Configuration Section (Drag & Drop) -->
                    <div id="harmonizerConfigSection" class="hidden">
                        <div id="harmonizerFileList" class="space-y-4"></div>
                        
                        <div id="harmonizerJoinOrder" class="mt-6">
                            <h3 class="text-lg font-semibold text-gray-700">Table Relationships</h3>
                            <p class="text-sm text-gray-600 mb-2">Drag and drop files to set relationship order for schema analysis.</p>
                            <div id="harmonizerJoinOrderDropzone" class="dropzone" ondragover="event.preventDefault(); this.classList.add('dragover');" 
                                 ondragleave="this.classList.remove('dragover');" ondrop="dropFileForHarmonizer(event)">
                            </div>
                        </div>
                        # Add this section after the join order section and before the target system section
<div class="mt-4">
    <label class="block text-gray-700 font-semibold mb-2">Filter Column:</label>
    <p class="text-sm text-gray-600 mb-2">Select a column from the first table to filter data (e.g., PersonNationalID).</p>
    <select id="harmonizerFilterColumn" 
            class="w-full p-2 border rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
        <option value="">Select a column</option>
    </select>
</div>

                        <!-- Target System and Requirements -->
                        <div class="mt-6 grid grid-cols-2 gap-6">
                            <div class="form-group">
                                <label class="block text-gray-700 font-semibold mb-2">Target System:</label>
                                <select id="harmonizerTargetSystem" class="form-control">
                                    <option value="snowflake">Snowflake</option>
                                    <option value="bigquery">BigQuery</option>
                                    <option value="redshift">Redshift</option>
                                    <option value="postgresql">PostgreSQL</option>
                                    <option value="databricks">Databricks</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label class="block text-gray-700 font-semibold mb-2">Output Config Path:</label>
                                <input type="text" id="harmonizerOutputPath" class="form-control" placeholder="e.g., harmonized_schema.json" value="harmonized_schema.json">
                            </div>
                        </div>

                        <div class="form-group">
                            <label class="block text-gray-700 font-semibold mb-2">Mapping Requirements & Business Rules:</label>
                            <textarea id="harmonizerMappingRequirements" class="form-control" rows="4" placeholder="Describe any specific mapping rules, business logic, or transformation requirements...">- Standardize column names across tables
- Handle data type conversions appropriately
- Preserve all business relationships
- Optimize for analytical queries
- Include data validation rules</textarea>
                        </div>

                        <button class="btn" onclick="analyzeAndHarmonizeSchema()" id="harmonizerBtn">Analyze & Harmonize Schema</button>
                        <div id="harmonizerStatus" class="status" style="display: none;"></div>
                    </div>
                    
                    <!-- Results Section -->
                    <div id="harmonizerResults" class="hidden">
                        <div class="mt-6">
                            <h3 class="text-lg font-semibold text-gray-700 mb-4">Harmonization Results</h3>
                            <div id="harmonizerOutput" class="code-output">
                                <button class="copy-btn" onclick="copyCode('harmonizerOutput')">Copy</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- S3 & Data Lakes Tab -->
            <div id="s3-tab" class="tab-content">
                <div class="card">
                    <h3>☁️ S3 & Data Lake Operations <span class="model-badge">CodeLlama</span></h3>
                    
                    <div class="form-group">
                        <label>Operation Type:</label>
                        <select id="s3Operation" class="form-control" onchange="toggleS3Fields()">
                            <option value="list">List Files in Bucket</option>
                            <option value="read" selected>Read Parquet/JSON Files</option>
                            <option value="upload">Upload Data to S3</option>
                            <option value="transform">Transform File Format</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="s3Bucket">S3 Bucket:</label>
                        <input type="text" id="s3Bucket" class="form-control" placeholder="e.g., my-data-bucket" value="company-data-lake">
                    </div>

                    <div class="form-group" id="s3PathGroup">
                        <label for="s3Path">S3 Path/Prefix:</label>
                        <input type="text" id="s3Path" class="form-control" placeholder="e.g., raw/customer_data/" value="raw/orders/">
                    </div>

                    <div class="form-group" id="fileFormatGroup">
                        <label for="fileFormat">File Format:</label>
                        <select id="fileFormat" class="form-control">
                            <option value="parquet" selected>Parquet</option>
                            <option value="json">JSON</option>
                            <option value="csv">CSV</option>
                            <option value="avro">Avro</option>
                        </select>
                    </div>

                    <button class="btn" onclick="generateS3Code()" id="s3Btn">Generate Code</button>
                    <div id="s3Status" class="status" style="display: none;"></div>
                    <div id="s3Output" class="code-output" style="display: none;">
                        <button class="copy-btn" onclick="copyCode('s3Output')">Copy</button>
                    </div>
                </div>
            </div>

            <!-- AI Chat Assistant Tab -->
            <div id="chat-tab" class="tab-content">
                <div class="card">
                    <h3>💬 Data Engineering AI Assistant <span class="model-badge">Gemma3</span></h3>
                    <div id="chatMessages" style="height: 400px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #f8fafc;">
                        <div class="message ai-message">
                            <strong>AI Assistant (Gemma3):</strong> Hello! I'm your DataGenie AI Co-Pilot. I can help you with:
                            <br>• SQL query generation (SQLCoder)
                            <br>• dbt test creation (SQLCoder)  
                            <br>• Pipeline design (CodeLlama)
                            <br>• Schema mapping & harmonization (SQLCoder)
                            <br>• Data quality checks (SQLCoder)
                            <br>• S3 operations (CodeLlama)
                            <br><br>What would you like to work on today?
                        </div>
                    </div>
                    <div class="form-group">
                        <textarea id="chatInput" class="form-control" placeholder="Ask me anything about data engineering...">How can I optimize my Snowflake queries?</textarea>
                    </div>
                    <button class="btn" onclick="sendChatMessage()" id="chatBtn">Send Message</button>
                </div>
            </div>

            <!-- Data Quality Tab -->
            <div id="quality-tab" class="tab-content">
                <div class="card">
                    <h3>🔍 Data Quality & Monitoring <span class="model-badge">SQLCoder</span></h3>
                    <div class="form-group">
                        <label for="qualityDataset">Dataset to Analyze:</label>
                        <input type="text" id="qualityDataset" class="form-control" value="customer_orders" placeholder="e.g., customer_orders">
                    </div>
                    <div class="form-group">
                        <label for="qualityChecks">Quality Checks Needed:</label>
                        <textarea id="qualityChecks" class="form-control" placeholder="Describe the data quality checks you need...">Check for null values, validate date ranges, ensure foreign key relationships, detect outliers in amounts</textarea>
                    </div>
                    <button class="btn" onclick="generateQualityChecks()" id="qualityBtn">Generate Quality Checks</button>
                    <div id="qualityStatus" class="status" style="display: none;"></div>
                    <div id="qualityOutput" class="code-output" style="display: none;">
                        <button class="copy-btn" onclick="copyCode('qualityOutput')">Copy</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // API Configuration
            const API_BASE_URL = window.location.origin;
            const ENDPOINTS = {
                GENERATE_SQL: `${API_BASE_URL}/ai/sql/generate`,
                GENERATE_DBT: `${API_BASE_URL}/ai/dbt/generate-tests`,
                GENERATE_PIPELINE: `${API_BASE_URL}/ai/pipeline/generate`,
                MAP_SCHEMAS: `${API_BASE_URL}/ai/schema/map`,
                HARMONIZE_SCHEMAS: `${API_BASE_URL}/ai/schema/harmonize`,
                GENERATE_S3: `${API_BASE_URL}/ai/s3/generate-code`,
                GENERATE_QUALITY: `${API_BASE_URL}/ai/quality/generate-checks`,
                CHAT: `${API_BASE_URL}/ai/chat`,
                ADMIN_FILES: `${API_BASE_URL}/admin/files`
            };

            // Tab Management
            function openTab(tabName) {
                document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');
                event.currentTarget.classList.add('active');
            }

            // Utility Functions
            function showStatus(elementId, message, type = 'success') {
                const statusEl = document.getElementById(elementId);
                statusEl.textContent = message;
                statusEl.className = `status ${type}`;
                statusEl.style.display = 'block';
                setTimeout(() => statusEl.style.display = 'none', 5000);
            }

            function copyCode(elementId) {
                const codeElement = document.getElementById(elementId);
                const code = codeElement.textContent.replace('Copy', '').trim();
                navigator.clipboard.writeText(code).then(() => {
                    const statusId = elementId.replace('Output', 'Status');
                    showStatus(statusId, 'Code copied to clipboard!');
                });
            }

            function setLoading(buttonId, isLoading) {
                const button = document.getElementById(buttonId);
                if (isLoading) {
                    button.innerHTML = '<span class="loading"></span> Processing...';
                    button.disabled = true;
                } else {
                    const originalText = buttonId.includes('sql') ? 'Generate SQL' :
                                      buttonId.includes('dbt') ? 'Generate dbt Tests' :
                                      buttonId.includes('pipeline') ? 'Generate Pipeline' :
                                      buttonId.includes('mapper') ? 'Map Schemas' :
                                      buttonId.includes('harmonizer') ? 'Analyze & Harmonize Schema' :
                                      buttonId.includes('s3') ? 'Generate Code' :
                                      buttonId.includes('quality') ? 'Generate Quality Checks' : 'Send Message';
                    button.innerHTML = originalText;
                    button.disabled = false;
                }
            }

            // API Functions
            async function generateSQL() {
                const prompt = document.getElementById('sqlPrompt').value.trim();
                const schemaContext = document.getElementById('schemaContext').value.trim();
                const databaseType = document.getElementById('databaseType').value;
                
                if (!prompt) {
                    showStatus('sqlStatus', 'Please enter a description', 'error');
                    return;
                }

                setLoading('sqlBtn', true);
                
                try {
                    const response = await fetch(ENDPOINTS.GENERATE_SQL, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            prompt: prompt,
                            schema_context: schemaContext,
                            database_type: databaseType
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        const outputEl = document.getElementById('sqlOutput');
                        outputEl.textContent = data.sql;
                        outputEl.style.display = 'block';
                        showStatus('sqlStatus', `SQL generated successfully using ${data.model}!`);
                    } else {
                        showStatus('sqlStatus', `Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    showStatus('sqlStatus', `Network error: ${error.message}`, 'error');
                } finally {
                    setLoading('sqlBtn', false);
                }
            }

            async function generateDBTTests() {
                const tableName = document.getElementById('tableName').value.trim();
                const columns = document.getElementById('tableColumns').value.split(',').map(col => col.trim());
                const requirements = document.getElementById('testRequirements').value.trim();
                
                if (!tableName || columns.length === 0) {
                    showStatus('dbtStatus', 'Please provide table name and columns', 'error');
                    return;
                }

                setLoading('dbtBtn', true);
                
                try {
                    const response = await fetch(ENDPOINTS.GENERATE_DBT, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            table_name: tableName,
                            columns: columns,
                            requirements: requirements
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        const outputEl = document.getElementById('dbtOutput');
                        outputEl.textContent = data.yaml;
                        outputEl.style.display = 'block';
                        showStatus('dbtStatus', `dbt tests generated successfully using ${data.model}!`);
                    } else {
                        showStatus('dbtStatus', `Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    showStatus('dbtStatus', `Network error: ${error.message}`, 'error');
                } finally {
                    setLoading('dbtBtn', false);
                }
            }

            async function generatePipeline() {
                const description = document.getElementById('pipelineDescription').value.trim();
                const tools = Array.from(document.getElementById('pipelineTools').selectedOptions).map(opt => opt.value);
                
                if (!description) {
                    showStatus('pipelineStatus', 'Please enter a pipeline description', 'error');
                    return;
                }

                setLoading('pipelineBtn', true);
                
                try {
                    const response = await fetch(ENDPOINTS.GENERATE_PIPELINE, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            description: description,
                            tools: tools
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        const outputEl = document.getElementById('pipelineOutput');
                        outputEl.textContent = data.code;
                        outputEl.style.display = 'block';
                        showStatus('pipelineStatus', `Pipeline generated successfully using ${data.model}!`);
                    } else {
                        showStatus('pipelineStatus', `Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    showStatus('pipelineStatus', `Network error: ${error.message}`, 'error');
                } finally {
                    setLoading('pipelineBtn', false);
                }
            }

            async function mapSchemas() {
                const sourceSystem = document.getElementById('sourceSystem').value.trim();
                const targetSystem = document.getElementById('targetSystem').value.trim();
                const sourceSchema = document.getElementById('sourceSchema').value.trim();
                const targetSchema = document.getElementById('targetSchema').value.trim();
                const mappingRules = document.getElementById('mappingRules').value.trim();
                
                if (!sourceSchema || !targetSchema) {
                    showStatus('mapperStatus', 'Please provide both source and target schemas', 'error');
                    return;
                }

                try {
                    // Parse JSON to validate
                    const sourceSchemaObj = JSON.parse(sourceSchema);
                    const targetSchemaObj = JSON.parse(targetSchema);
                } catch (e) {
                    showStatus('mapperStatus', 'Invalid JSON in schema definitions', 'error');
                    return;
                }

                setLoading('mapperBtn', true);
                
                try {
                    const response = await fetch(ENDPOINTS.MAP_SCHEMAS, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            source_schema: JSON.parse(sourceSchema),
                            target_schema: JSON.parse(targetSchema),
                            source_system: sourceSystem,
                            target_system: targetSystem,
                            mapping_rules: mappingRules
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        const outputEl = document.getElementById('mapperOutput');
                        outputEl.textContent = data.mapping_plan;
                        outputEl.style.display = 'block';
                        showStatus('mapperStatus', `Schema mapping generated successfully using ${data.model}!`);
                    } else {
                        showStatus('mapperStatus', `Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    showStatus('mapperStatus', `Network error: ${error.message}`, 'error');
                } finally {
                    setLoading('mapperBtn', false);
                }
            }

            // Schema Harmonizer Functions
            let harmonizerFiles = [];
            let harmonizerJoinOrder = [];

            async function listFilesForHarmonizer() {
                const basePath = document.getElementById('harmonizerBasePath').value;
                if (!basePath) {
                    showStatus('harmonizerStatus', 'Please enter a data directory path.', 'error');
                    return;
                }
                
                try {
                    const response = await fetch(`${ENDPOINTS.ADMIN_FILES}?base_path=${encodeURIComponent(basePath)}`);
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(errorText);
                    }
                    harmonizerFiles = await response.json();
                    const fileCheckboxes = document.getElementById('harmonizerFileCheckboxes');
                    fileCheckboxes.innerHTML = '';
                    
                    if (harmonizerFiles.length === 0) {
                        fileCheckboxes.innerHTML = '<p class="text-gray-600">No CSV files found in the specified directory.</p>';
                    } else {
                        harmonizerFiles.forEach((file, index) => {
                            const div = document.createElement('div');
                            div.innerHTML = `
                                <label class="flex items-center space-x-2">
                                    <input type="checkbox" id="harmonizer_file_${index}" value="${file.name}" class="h-4 w-4">
                                    <span>${file.name}.csv (Columns: ${file.columns.join(', ')})</span>
                                </label>
                            `;
                            fileCheckboxes.appendChild(div);
                        });
                    }
                    
                    document.getElementById('harmonizerFileSelection').classList.remove('hidden');
                    document.getElementById('harmonizerConfigSection').classList.add('hidden');
                    document.getElementById('harmonizerResults').classList.add('hidden');
                    
                } catch (error) {
                    showStatus('harmonizerStatus', 'Error: ' + error.message, 'error');
                }
            }

            function configureFilesForHarmonizer() {
    const selectedFiles = Array.from(document.querySelectorAll('#harmonizerFileCheckboxes input:checked'))
        .map(input => harmonizerFiles.find(file => file.name === input.value));
        
    if (selectedFiles.length === 0) {
        showStatus('harmonizerStatus', 'Please select at least one file.', 'error');
        return;
    }
    
    harmonizerJoinOrder = selectedFiles.map(file => file.name);
    const fileListDiv = document.getElementById('harmonizerFileList');
    fileListDiv.innerHTML = '';
    const joinOrderDropzone = document.getElementById('harmonizerJoinOrderDropzone');
    joinOrderDropzone.innerHTML = '';
    
    // ✅ Populate filter column dropdown with columns from the first file
    const firstFile = selectedFiles.find(file => file.name === harmonizerJoinOrder[0]);
    let firstFileColumns = Array.isArray(firstFile.columns) ? firstFile.columns : firstFile.columns.split(",").map(col => col.trim());
    const filterColumnSelect = document.getElementById('harmonizerFilterColumn');
    filterColumnSelect.innerHTML = '<option value="">Select a column</option>' +
        firstFileColumns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    // Check for column conflicts
    const columnCounts = {};
    selectedFiles.forEach(file => {
        let fileColumns = Array.isArray(file.columns) ? file.columns : file.columns.split(",").map(col => col.trim());
        fileColumns.forEach(col => {
            columnCounts[col] = (columnCounts[col] || 0) + 1;
        });
    });
    
    const conflicts = Object.keys(columnCounts).filter(col => columnCounts[col] > 1);
    const conflictWarning = document.getElementById('harmonizerConflictWarning');
    if (!conflictWarning) {
        // Create conflict warning element if it doesn't exist
        const warningDiv = document.createElement('div');
        warningDiv.id = 'harmonizerConflictWarning';
        warningDiv.className = 'conflict-warning';
        document.getElementById('harmonizerConfigSection').insertBefore(warningDiv, document.getElementById('harmonizerJoinOrder'));
    }
    
    if (conflicts.length > 0) {
        document.getElementById('harmonizerConflictWarning').innerText = `Warning: Columns ${conflicts.join(', ')} appear in multiple files and may cause conflicts.`;
        document.getElementById('harmonizerConflictWarning').classList.remove('hidden');
    } else {
        document.getElementById('harmonizerConflictWarning').classList.add('hidden');
    }
    
    selectedFiles.forEach((file, index) => {
        let columns = Array.isArray(file.columns) ? file.columns : file.columns.split(",").map(col => col.trim());
        const fileDiv = document.createElement('div');
        fileDiv.className = 'bg-gray-50 p-4 rounded-md border';
        fileDiv.innerHTML = `
            <h3 class="text-lg font-semibold text-gray-700 mb-2">${file.name}</h3>
            <label class="block text-gray-600">Primary Key (optional):</label>
            <p class="text-sm text-gray-600 mb-2">Column identifying unique records, recommended for the first file.</p>
            <select id="harmonizer_primaryKey_${index}" 
                    class="w-full p-2 border rounded-md mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="">None</option>
                ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
            </select>
            <label class="block text-gray-600">Join Relationships:</label>
            <p class="text-sm text-gray-600 mb-2">Define how this table connects to others.</p>
            <div class="grid grid-cols-2 gap-2">
                <div>
                    <label class="block text-sm text-gray-600 mb-1">Join Key In:</label>
                    <select id="harmonizer_joinKeyIn_${index}" 
                            class="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="">No Join In</option>
                        ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
                <div>
                    <label class="block text-sm text-gray-600 mb-1">Join Key Out:</label>
                    <select id="harmonizer_joinKeyOut_${index}" 
                            class="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="">No Join Out</option>
                        ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                    </select>
                </div>
            </div>
        `;
        fileListDiv.appendChild(fileDiv);
        
        const dragItem = document.createElement('div');
        dragItem.className = 'draggable';
        dragItem.draggable = true;
        dragItem.id = `harmonizer_drag_${file.name}`;
        dragItem.innerText = file.name;
        dragItem.ondragstart = (ev) => ev.dataTransfer.setData('text/plain', file.name);
        joinOrderDropzone.appendChild(dragItem);
    });
    
    document.getElementById('harmonizerFileSelection').classList.add('hidden');
    document.getElementById('harmonizerConfigSection').classList.remove('hidden');
    document.getElementById('harmonizerResults').classList.add('hidden');
}

           function dropFileForHarmonizer(event) {
    event.preventDefault();
    event.target.classList.remove('dragover');
    const fileName = event.dataTransfer.getData('text/plain');
    const index = harmonizerJoinOrder.indexOf(fileName);
    if (index !== -1) {
        harmonizerJoinOrder.splice(index, 1);
    }
    const dropzone = document.getElementById('harmonizerJoinOrderDropzone');
    const draggedElement = document.getElementById(`harmonizer_drag_${fileName}`);
    dropzone.appendChild(draggedElement);
    const items = Array.from(dropzone.children).map(item => item.innerText);
    harmonizerJoinOrder = items;

    // ✅ Update filter column dropdown when join order changes
    const firstFile = harmonizerFiles.find(file => file.name === harmonizerJoinOrder[0]);
    if (firstFile) {
        let firstFileColumns = Array.isArray(firstFile.columns) ? firstFile.columns : firstFile.columns.split(",").map(col => col.trim());
        const filterColumnSelect = document.getElementById('harmonizerFilterColumn');
        filterColumnSelect.innerHTML = '<option value="">Select a column</option>' +
            firstFileColumns.map(col => `<option value="${col}">${col}</option>`).join('');
    }
}

            async function analyzeAndHarmonizeSchema() {
    const basePath = document.getElementById('harmonizerBasePath').value;
    const targetSystem = document.getElementById('harmonizerTargetSystem').value;
    const outputPath = document.getElementById('harmonizerOutputPath').value;
    const mappingRequirements = document.getElementById('harmonizerMappingRequirements').value;
    const filterColumn = document.getElementById('harmonizerFilterColumn').value; // Get filter column
    
    if (!basePath || harmonizerJoinOrder.length === 0 || !filterColumn) {
        showStatus('harmonizerStatus', 'Please configure file relationships and select a filter column first.', 'error');
        return;
    }

    // Build files configuration
    const files = [];
    document.querySelectorAll('#harmonizerFileList .bg-gray-50').forEach((_, index) => {
        const fileName = harmonizerJoinOrder[index];
        const fileInfo = {
            name: fileName,
            path: harmonizerFiles.find(f => f.name === fileName).path
        };
        
        const primaryKey = document.getElementById(`harmonizer_primaryKey_${index}`).value;
        const joinKeyIn = document.getElementById(`harmonizer_joinKeyIn_${index}`).value;
        const joinKeyOut = document.getElementById(`harmonizer_joinKeyOut_${index}`).value;
        
        if (primaryKey) fileInfo.primary_key = primaryKey;
        if (joinKeyIn) fileInfo.join_key_in = joinKeyIn;
        if (joinKeyOut) fileInfo.join_key_out = joinKeyOut;
        
        files.push(fileInfo);
    });

    const config = {
        path: basePath,
        files: files,
        join_order: harmonizerJoinOrder,
        filter_column: filterColumn  // Include filter_column in config
    };

    setLoading('harmonizerBtn', true);
    
    try {
        // ✅ FIRST: Save the configuration using save-config endpoint
        const saveResponse = await fetch('/admin/save-config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                base_path: basePath,
                files: files,
                join_order: harmonizerJoinOrder,
                output_config_path: outputPath,
                filter_column: filterColumn
            })
        });

        if (!saveResponse.ok) {
            const errorText = await saveResponse.text();
            throw new Error(`Failed to save config: ${errorText}`);
        }

        const saveResult = await saveResponse.json();
        console.log('✅ Config saved:', saveResult);

        // ✅ SECOND: Now call the schema harmonization endpoint
        const response = await fetch(ENDPOINTS.HARMONIZE_SCHEMAS, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                config: config,
                target_system: targetSystem,
                mapping_requirements: mappingRequirements,
                output_config_path: outputPath
            })
        });

        const data = await response.json();
        
        if (data.success) {
            const outputEl = document.getElementById('harmonizerOutput');
            outputEl.textContent = data.harmonization_plan;
            document.getElementById('harmonizerResults').classList.remove('hidden');
            let successMsg = `Schema harmonized successfully using ${data.model}!`;
            if (data.config_saved) {
                successMsg += ` Configuration saved to ${outputPath}`;
            }
            showStatus('harmonizerStatus', successMsg);
        } else {
            showStatus('harmonizerStatus', `Error: ${data.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showStatus('harmonizerStatus', `Error: ${error.message}`, 'error');
    } finally {
        setLoading('harmonizerBtn', false);
    }
}
            // Toggle S3 fields based on operation type
            function toggleS3Fields() {
                const operation = document.getElementById('s3Operation').value;
                const pathGroup = document.getElementById('s3PathGroup');
                const formatGroup = document.getElementById('fileFormatGroup');
                
                if (operation === 'list') {
                    pathGroup.style.display = 'block';
                    formatGroup.style.display = 'none';
                } else {
                    pathGroup.style.display = 'block';
                    formatGroup.style.display = 'block';
                }
            }

            // Generate S3 Code
            async function generateS3Code() {
                const operation = document.getElementById('s3Operation').value;
                const bucket = document.getElementById('s3Bucket').value.trim();
                const path = document.getElementById('s3Path').value.trim();
                const fileFormat = document.getElementById('fileFormat').value;
                
                if (!bucket) {
                    showStatus('s3Status', 'Please enter an S3 bucket name', 'error');
                    return;
                }

                setLoading('s3Btn', true);
                
                try {
                    const response = await fetch(ENDPOINTS.GENERATE_S3, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            operation: operation,
                            bucket: bucket,
                            path: path,
                            file_format: fileFormat
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        const outputEl = document.getElementById('s3Output');
                        outputEl.textContent = data.code;
                        outputEl.style.display = 'block';
                        showStatus('s3Status', `S3 code generated successfully using ${data.model}!`);
                    } else {
                        showStatus('s3Status', `Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    showStatus('s3Status', `Network error: ${error.message}`, 'error');
                } finally {
                    setLoading('s3Btn', false);
                }
            }

            // Generate Quality Checks
            async function generateQualityChecks() {
                const dataset = document.getElementById('qualityDataset').value.trim();
                const checks = document.getElementById('qualityChecks').value.trim();
                const framework = 'dbt'; // Default framework
                
                if (!dataset || !checks) {
                    showStatus('qualityStatus', 'Please provide dataset and quality checks', 'error');
                    return;
                }

                setLoading('qualityBtn', true);
                
                try {
                    const response = await fetch(ENDPOINTS.GENERATE_QUALITY, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            dataset: dataset,
                            checks: checks,
                            framework: framework
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        const outputEl = document.getElementById('qualityOutput');
                        outputEl.textContent = data.checks;
                        outputEl.style.display = 'block';
                        showStatus('qualityStatus', `Quality checks generated successfully using ${data.model}!`);
                    } else {
                        showStatus('qualityStatus', `Error: ${data.detail || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    showStatus('qualityStatus', `Network error: ${error.message}`, 'error');
                } finally {
                    setLoading('qualityBtn', false);
                }
            }

            // Chat Functionality
            async function sendChatMessage() {
                const message = document.getElementById('chatInput').value.trim();
                const chatMessages = document.getElementById('chatMessages');
                
                if (!message) {
                    return;
                }

                // Add user message
                const userMessage = document.createElement('div');
                userMessage.className = 'message user-message';
                userMessage.innerHTML = `<strong>You:</strong> ${message}`;
                chatMessages.appendChild(userMessage);
                
                // Clear input
                document.getElementById('chatInput').value = '';
                
                setLoading('chatBtn', true);
                
                try {
                    const response = await fetch(ENDPOINTS.CHAT, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: message,
                            context: 'data engineering assistance'
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        // Add AI response
                        const aiMessage = document.createElement('div');
                        aiMessage.className = 'message ai-message';
                        aiMessage.innerHTML = `<strong>AI Assistant (${data.model}):</strong> ${data.response}`;
                        chatMessages.appendChild(aiMessage);
                        
                        // Scroll to bottom
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    } else {
                        const errorMessage = document.createElement('div');
                        errorMessage.className = 'message ai-message';
                        errorMessage.innerHTML = `<strong>AI Assistant:</strong> Error: ${data.detail || 'Unknown error'}`;
                        chatMessages.appendChild(errorMessage);
                    }
                } catch (error) {
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'message ai-message';
                    errorMessage.innerHTML = `<strong>AI Assistant:</strong> Network error: ${error.message}`;
                    chatMessages.appendChild(errorMessage);
                } finally {
                    setLoading('chatBtn', false);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }

            // Handle Enter key in chat
            document.addEventListener('DOMContentLoaded', function() {
                const chatInput = document.getElementById('chatInput');
                if (chatInput) {
                    chatInput.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendChatMessage();
                        }
                    });
                }
                // Initialize S3 fields
                toggleS3Fields();
            });
        </script>
    </body>
    </html>
    """
    return html_content

def get_admin_ui():
    """Return the Admin UI HTML"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DataGenie Admin</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .btn { background: #2563eb; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DataGenie Admin Panel</h1>
            
            <div class="card">
                <h2>File Management</h2>
                <div class="form-group">
                    <label>Base Path:</label>
                    <input type="text" id="basePath" placeholder="/path/to/your/data">
                </div>
                <button class="btn" onclick="listFiles()">List CSV Files</button>
                <div id="fileList"></div>
            </div>

            <div class="card">
                <h2>Configuration Builder</h2>
                <div id="configBuilder">
                    <div class="form-group">
                        <label>Output Config Path:</label>
                        <input type="text" id="outputConfigPath" value="config.json">
                    </div>
                    <div class="form-group">
                        <label>Filter Column:</label>
                        <input type="text" id="filterColumn" value="PersonNationalID">
                    </div>
                    <button class="btn" onclick="saveConfig()">Save Configuration</button>
                </div>
            </div>

            <div class="card">
                <h2>AI Co-Pilot Tools</h2>
                <p>Access the full AI Co-Pilot interface for advanced data engineering tasks:</p>
                <a href="/ui" class="btn">Open AI Co-Pilot</a>
            </div>
        </div>

        <script>
            async function listFiles() {
                const basePath = document.getElementById('basePath').value;
                if (!basePath) {
                    alert('Please enter a base path');
                    return;
                }

                try {
                    const response = await fetch(`/admin/files?base_path=${encodeURIComponent(basePath)}`);
                    const files = await response.json();
                    
                    const fileList = document.getElementById('fileList');
                    fileList.innerHTML = '<h3>Found Files:</h3>';
                    
                    files.forEach(file => {
                        const fileDiv = document.createElement('div');
                        fileDiv.className = 'file-item';
                        fileDiv.innerHTML = `
                            <strong>${file.name}</strong> - ${file.path}
                            <br><small>Columns: ${file.columns.join(', ')}</small>
                        `;
                        fileList.appendChild(fileDiv);
                    });
                } catch (error) {
                    alert('Error listing files: ' + error.message);
                }
            }

            async function saveConfig() {
                // Implementation for saving configuration
                alert('Configuration save functionality would be implemented here');
            }
        </script>
    </body>
    </html>
    """

def get_prompts_management_ui():
    """Return the prompts management HTML interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prompt Management - DataGenie</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js"></script>
        <style>
            .variable { 
                background-color: #e0f2fe; 
                color: #0369a1; 
                padding: 2px 6px; 
                border-radius: 4px; 
                font-family: 'Courier New', monospace; 
                font-weight: bold;
                font-size: 0.9em;
            }
            .prompt-editor { 
                height: 300px; 
                width: 100%; 
                border: 1px solid #d1d5db; 
                border-radius: 6px; 
            }
            .tab-button { 
                padding: 12px 24px; 
                border: none; 
                background: #f3f4f6; 
                cursor: pointer; 
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            .tab-button.active { 
                background: white; 
                border-bottom-color: #3b82f6; 
                color: #3b82f6;
                font-weight: 600;
            }
            .tab-button:hover {
                background: #e5e7eb;
            }
        </style>
    </head>
    <body class="bg-gray-50 min-h-screen p-6">
        <div class="max-w-6xl mx-auto bg-white rounded-xl shadow-lg p-8">
            <div class="flex justify-between items-center mb-8 pb-6 border-b border-gray-200">
                <div>
                    <h1 class="text-3xl font-bold text-gray-800">🔧 Prompt Management</h1>
                    <p class="text-gray-600 mt-2">Manage authentication prompts for different verification scenarios</p>
                </div>
                <div class="flex space-x-4">
                    <a href="/ui" class="bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 transition duration-200 font-medium">
                        ← Back to AI Co-Pilot
                    </a>
                    <a href="/admin" class="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 transition duration-200 font-medium">
                        ⚙️ Admin Panel
                    </a>
                </div>
            </div>

            <div class="mb-8 p-6 bg-blue-50 rounded-xl border border-blue-200">
                <h2 class="text-xl font-semibold text-blue-800 mb-3">About Prompt Management</h2>
                <p class="text-blue-700 mb-3">
                    Customize the authentication questions generated by the system. Variables like 
                    <span class="variable">{target_id}</span> and <span class="variable">{data_str}</span> 
                    are automatically replaced during execution and cannot be modified.
                </p>
                <div class="text-sm text-blue-600">
                    <p><strong>💡 Tip:</strong> Keep questions natural and conversational - they'll be used by agents during verification calls.</p>
                </div>
            </div>

            <div id="loadingMessage" class="p-6 bg-yellow-50 rounded-lg border border-yellow-200 mb-6">
                <div class="flex items-center">
                    <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-yellow-600 mr-3"></div>
                    <p class="text-yellow-700 font-medium">Loading prompts...</p>
                </div>
            </div>

            <div id="promptsContainer" class="hidden">
                <!-- Tabs for different prompt types -->
                <div class="border-b border-gray-200 mb-8">
                    <div id="promptTabs" class="flex space-x-1 overflow-x-auto"></div>
                </div>

                <!-- Prompt Editor -->
                <div id="promptEditor" class="hidden">
                    <div class="mb-6 p-6 bg-gray-50 rounded-xl border border-gray-200">
                        <h3 id="currentPromptName" class="text-2xl font-bold text-gray-800 mb-2"></h3>
                        <p id="currentPromptDescription" class="text-gray-600 text-lg"></p>
                    </div>

                    <div class="mb-6 p-6 bg-green-50 rounded-xl border border-green-200">
                        <h4 class="font-semibold text-green-800 text-lg mb-3">🛠️ Available Variables:</h4>
                        <p class="text-green-700 mb-3">These variables will be automatically replaced with actual values:</p>
                        <div id="variableList" class="flex flex-wrap gap-3"></div>
                    </div>

                    <div class="mb-6">
                        <label class="block text-gray-700 font-semibold text-lg mb-3">Prompt Template:</label>
                        <div class="mb-2 text-sm text-gray-600">
                            This template generates 3 verification questions. Use natural, conversational language.
                        </div>
                        <div id="editor" class="prompt-editor border-2 border-gray-300 rounded-lg"></div>
                    </div>

                    <div class="flex space-x-4">
                        <button onclick="savePrompt()" class="bg-green-500 text-white px-8 py-3 rounded-lg hover:bg-green-600 transition duration-200 font-medium flex items-center">
                            💾 Save Changes
                        </button>
                        <button onclick="resetPrompt()" class="bg-yellow-500 text-white px-8 py-3 rounded-lg hover:bg-yellow-600 transition duration-200 font-medium flex items-center">
                            🔄 Reset to Default
                        </button>
                        <button onclick="previewPrompt()" class="bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 transition duration-200 font-medium flex items-center">
                            👁️ Preview Prompt
                        </button>
                    </div>

                    <div id="saveStatus" class="mt-6"></div>
                </div>

                <!-- Preview Modal -->
                <div id="previewModal" class="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50 p-4">
                    <div class="bg-white rounded-xl p-8 max-w-4xl w-full max-h-[90vh] overflow-auto">
                        <div class="flex justify-between items-center mb-6 pb-4 border-b border-gray-200">
                            <h3 class="text-2xl font-bold text-gray-800">Prompt Preview</h3>
                            <button onclick="closePreview()" class="text-gray-500 hover:text-gray-700 text-2xl font-bold">×</button>
                        </div>
                        <div class="mb-4 p-4 bg-gray-100 rounded-lg border">
                            <h4 class="font-semibold text-gray-700 mb-2">Formatted Prompt:</h4>
                            <div id="previewContent" class="font-mono whitespace-pre-wrap text-sm bg-white p-4 rounded border"></div>
                        </div>
                        <div class="text-sm text-gray-600 bg-blue-50 p-4 rounded-lg">
                            <p class="font-semibold">💡 Note:</p>
                            <p>Variables will be replaced with actual values during execution. The LLM will generate 3 verification questions based on this template.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let allPrompts = {};
            let currentPromptType = '';
            let editor = null;

            // Load prompts on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadPrompts();
                
                // Initialize ACE editor
                editor = ace.edit("editor");
                editor.setTheme("ace/theme/chrome");
                editor.session.setMode("ace/mode/text");
                editor.setOptions({
                    fontSize: "14px",
                    showPrintMargin: false,
                    wrap: true,
                    showLineNumbers: true,
                    showGutter: true
                });
            });

            async function loadPrompts() {
                try {
                    const response = await fetch('/api/prompts');
                    const data = await response.json();
                    allPrompts = data.prompts;
                    
                    displayPromptTabs();
                    document.getElementById('loadingMessage').classList.add('hidden');
                    document.getElementById('promptsContainer').classList.remove('hidden');
                    
                    // Load first prompt by default
                    if (Object.keys(allPrompts).length > 0) {
                        const firstType = Object.keys(allPrompts)[0];
                        loadPrompt(firstType);
                    }
                } catch (error) {
                    document.getElementById('loadingMessage').innerHTML = 
                        '<div class="p-4 bg-red-100 text-red-700 rounded-lg border border-red-200">❌ Error loading prompts: ' + error.message + '</div>';
                }
            }

            function displayPromptTabs() {
                const tabsContainer = document.getElementById('promptTabs');
                tabsContainer.innerHTML = '';
                
                Object.keys(allPrompts).forEach(promptType => {
                    const prompt = allPrompts[promptType];
                    const button = document.createElement('button');
                    button.className = 'tab-button';
                    button.textContent = prompt.name || promptType;
                    button.onclick = () => loadPrompt(promptType);
                    
                    tabsContainer.appendChild(button);
                });
            }

            function loadPrompt(promptType) {
                currentPromptType = promptType;
                const prompt = allPrompts[promptType];
                
                // Update active tab
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                event.currentTarget.classList.add('active');
                
                // Update prompt info
                document.getElementById('currentPromptName').textContent = prompt.name || promptType;
                document.getElementById('currentPromptDescription').textContent = prompt.description || '';
                
                // Display variables
                const variableList = document.getElementById('variableList');
                variableList.innerHTML = '';
                if (prompt.variables) {
                    prompt.variables.forEach(variable => {
                        const span = document.createElement('span');
                        span.className = 'variable';
                        span.textContent = `{${variable}}`;
                        variableList.appendChild(span);
                    });
                }
                
                // Set editor content
                editor.setValue(prompt.prompt || '', -1);
                
                // Show editor
                document.getElementById('promptEditor').classList.remove('hidden');
            }

            async function savePrompt() {
                try {
                    const updatedPrompt = {
                        name: allPrompts[currentPromptType].name,
                        description: allPrompts[currentPromptType].description,
                        prompt: editor.getValue()
                    };
                    
                    const response = await fetch(`/api/prompts/${currentPromptType}`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(updatedPrompt)
                    });
                    
                    const result = await response.json();
                    
                    const statusDiv = document.getElementById('saveStatus');
                    if (result.status === 'success') {
                        statusDiv.innerHTML = '<div class="p-4 bg-green-100 text-green-700 rounded-lg border border-green-200">✅ ' + result.message + '</div>';
                        
                        // Reload prompts to get updated data
                        await loadPrompts();
                    } else {
                        statusDiv.innerHTML = '<div class="p-4 bg-red-100 text-red-700 rounded-lg border border-red-200">❌ ' + result.message + '</div>';
                    }
                    
                    // Auto-hide status after 5 seconds
                    setTimeout(() => {
                        statusDiv.innerHTML = '';
                    }, 5000);
                } catch (error) {
                    document.getElementById('saveStatus').innerHTML = 
                        '<div class="p-4 bg-red-100 text-red-700 rounded-lg border border-red-200">❌ Error: ' + error.message + '</div>';
                }
            }

            async function resetPrompt() {
                if (!confirm('Are you sure you want to reset this prompt to its default value? This cannot be undone.')) {
                    return;
                }
                
                try {
                    const response = await fetch(`/api/prompts/reset/${currentPromptType}`, {
                        method: 'POST'
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        alert('✅ ' + result.message);
                        await loadPrompts();
                    } else {
                        alert('❌ ' + result.message);
                    }
                } catch (error) {
                    alert('❌ Error: ' + error.message);
                }
            }

            function previewPrompt() {
                const previewContent = document.getElementById('previewContent');
                const promptText = editor.getValue();
                
                // Highlight variables in preview
                let highlightedPrompt = promptText;
                if (allPrompts[currentPromptType] && allPrompts[currentPromptType].variables) {
                    allPrompts[currentPromptType].variables.forEach(variable => {
                        const regex = new RegExp(`\\{${variable}\\}`, 'g');
                        highlightedPrompt = highlightedPrompt.replace(regex, 
                            `<span class="variable">{${variable}}</span>`);
                    });
                }
                
                previewContent.innerHTML = highlightedPrompt;
                document.getElementById('previewModal').classList.remove('hidden');
            }

            function closePreview() {
                document.getElementById('previewModal').classList.add('hidden');
            }

            // Close modal when clicking outside
            document.getElementById('previewModal').addEventListener('click', function(e) {
                if (e.target.id === 'previewModal') {
                    closePreview();
                }
            });
        </script>
    </body>
    </html>
    """

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )