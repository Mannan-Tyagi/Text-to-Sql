import json
import time
import re
import psycopg2
from psycopg2 import sql
import os
from llama_index.llms.groq import Groq   # Import Groq LLM

# Global counter for token usage
total_tokens = 0

class NLSQLAgent:
    """
    Agent for generating and correcting SQL queries using Groq's language models.
    """
    
    def __init__(self, db_config, api_key, model="llama3-70b-8192"):
        """
        Initialize the NL to SQL Agent.
        
        :param db_config: Dictionary with database configuration (dbname, user, password, host, port)
        :param api_key: Groq API key
        :param model: Model name to use (default is llama3-70b-8192)
        """
        self.db_config = db_config
        self.api_key = api_key
        self.model = model
        # Create a Groq LLM instance using the provided model and API key
        self.llm = Groq(model=self.model, api_key=self.api_key)
        self.schema = self._get_database_schema()
        
    def _get_database_schema(self):
        """
        Get the schema of the database including tables, columns, and relationships.
        
        :return: Dictionary containing the database schema.
        """
        try:
            conn = psycopg2.connect(
                dbname=self.db_config['dbname'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', '5432')
            )
            cursor = conn.cursor()
            
            # Get all tables in public schema
            cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema='public'
            """)
            tables = cursor.fetchall()
            
            schema = {}
            for table in tables:
                table_name = table[0]
                
                # Get columns
                cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                """)
                columns = cursor.fetchall()
                
                # Get primary keys
                cursor.execute(f"""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                    AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = '{table_name}'::regclass
                AND    i.indisprimary
                """)
                primary_keys = [pk[0] for pk in cursor.fetchall()]
                
                # Get foreign keys
                cursor.execute(f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                      AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name='{table_name}'
                """)
                foreign_keys = cursor.fetchall()
                
                # Get sample data (first 5 rows) for debugging purposes
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                    sample_rows = cursor.fetchall()
                    sample_columns = [desc[0] for desc in cursor.description]
                    sample_data = [dict(zip(sample_columns, row)) for row in sample_rows]
                except Exception as e:
                    print(f"Error getting sample data for {table_name}: {e}")
                    sample_data = []
                
                schema[table_name] = {
                    "columns": [{"name": col[0], "type": col[1], "nullable": col[2]} for col in columns],
                    "primary_keys": primary_keys,
                    "foreign_keys": [{"column": fk[0], "references_table": fk[1], "references_column": fk[2]} for fk in foreign_keys],
                    "sample_data": sample_data[:2]  # Keep only 2 samples to reduce prompt size
                }
            
            cursor.close()
            conn.close()
            return schema
            
        except Exception as e:
            print(f"Error getting database schema: {e}")
            return {}
    
    def _build_concise_schema_summary(self):
        """
        Build a concise schema summary listing each table and its columns with types.
        This summary is designed to reduce token usage in the prompt.
        
        :return: A string representing the concise schema summary.
        """
        lines = []
        # Optionally limit the number of tables to include if the schema is too large.
        tables_to_include = list(self.schema.keys())[:20]  # include only first 20 tables
        for table in tables_to_include:
            info = self.schema.get(table, {})
            col_list = [f"{col['name']} ({col['type']})" for col in info.get("columns", [])]
            line = f"{table}: " + ", ".join(col_list)
            lines.append(line)
        return "\n".join(lines)
    
    def _call_groq_api(self, messages, temperature=0.0, max_tokens=1000, n=1):
        """
        Call the Groq LLM to get a response.
        This function converts multi-turn messages into a single prompt string.
        
        :param messages: List of message dictionaries.
        :param temperature: Temperature for the model (unused with Groq LLM).
        :param max_tokens: Maximum tokens to generate (unused with Groq LLM).
        :param n: Number of responses to generate (unused with Groq LLM).
        :return: Dictionary with a 'choices' list formatted like a typical LLM response.
        """
        global total_tokens
        # Combine messages into one prompt
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n"
        
        try:
            response_obj = self.llm.complete(prompt)
            # Try to extract a string response; check for attribute 'text'
            if hasattr(response_obj, "text"):
                response_text = response_obj.text
            else:
                # Fallback: convert the response object to string
                response_text = str(response_obj)
            if not response_text:
                raise ValueError("Empty response from Groq LLM. Check your API key and network connectivity.")
        except Exception as e:
            # Check if error is due to token limits
            error_message = str(e)
            if "413" in error_message or "tokens per minute" in error_message:
                print("The request exceeded the token limit. Please reduce the prompt size and try again.")
            else:
                print(f"Error calling Groq API: {e}")
            # Return an error response with empty content.
            return {"choices": [{"message": {"content": ""}}]}
        
        # Groq LLM does not provide token usage details; total_tokens remains unchanged.
        return {"choices": [{"message": {"content": response_text.strip()}}]}
    
    def _extract_sql_from_text(self, text):
        """
        Extract SQL query from text that may contain extra explanations.
        Remove newlines so that the SQL query is in a single line.
        
        :param text: Text possibly containing a SQL query.
        :return: The extracted SQL query in one single line.
        """
        if not text:
            return ""
        
        # Attempt to extract SQL within triple backticks
        sql_pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(sql_pattern, text, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
            return " ".join(sql_query.split())
        
        # Attempt to extract SQL within single backticks
        sql_pattern = r"`(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH).*?`"
        match = re.search(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(0).strip('`')
            return " ".join(sql_query.split())
        
        # Attempt to extract any SQL-like pattern ending with a semicolon
        sql_pattern = r"(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)[\s\S]*?;"
        match = re.search(sql_pattern, text, re.IGNORECASE)
        if match:
            sql_query = match.group(0).strip()
            return " ".join(sql_query.split())
        
        # Fallback: remove newlines from the original text
        return " ".join(text.split())
    
    def generate_sql(self, nl_query):
        """
        Generate SQL query from a natural language query.
        
        :param nl_query: Natural language query.
        :return: Generated SQL query on one single line.
        """
        if not nl_query.strip():
            return ""
            
        # Build a concise schema summary to reduce the token usage.
        concise_schema = self._build_concise_schema_summary()
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert SQL query generator specialized in PostgreSQL. Your task is to convert "
                    "natural language queries into accurate SQL queries.\n\n"
                    "Here is the database schema (concise):\n" + concise_schema + "\n\n"
                    "Guidelines:\n"
                    "1. Generate syntactically correct PostgreSQL queries.\n"
                    "2. Use the exact column and table names from the schema.\n"
                    "3. Use proper join conditions based on foreign key relationships.\n"
                    "4. Optimize queries for performance.\n"
                    "5. Return ONLY the SQL query with no explanation or markdown.\n"
                    "6. Ensure the query addresses all requirements in the natural language request."
                )
            },
            {
                "role": "user",
                "content": f"Convert this natural language query to SQL:\n\n{nl_query}"
            }
        ]
        
        print(f"Generating SQL for: {nl_query[:50]}...")
        response = self._call_groq_api(messages)
        if 'choices' in response and len(response['choices']) > 0:
            sql_query = self._extract_sql_from_text(response['choices'][0]['message']['content'])
            return sql_query
        else:
            print(f"Error in generate_sql response: {response}")
            return ""
    
    def correct_sql(self, incorrect_query):
        """
        Correct an incorrect SQL query.
        
        :param incorrect_query: Incorrect SQL query.
        :return: Corrected SQL query in one single line.
        """
        if not incorrect_query.strip():
            return ""
            
        concise_schema = self._build_concise_schema_summary()
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert SQL query debugger and corrector. Your task is to fix incorrect SQL queries "
                    "based on PostgreSQL syntax and the database schema.\n\n"
                    "Here is the database schema (concise):\n" + concise_schema + "\n\n"
                    "Common issues to fix:\n"
                    "1. Syntax errors.\n"
                    "2. Incorrect table or column names.\n"
                    "3. Wrong join conditions.\n"
                    "4. Missing quotes around string literals.\n"
                    "5. Improper data type handling.\n"
                    "6. Invalid operators or expressions.\n\n"
                    "Return ONLY the corrected SQL query with no explanation or markdown."
                )
            },
            {
                "role": "user",
                "content": f"Fix this SQL query:\n\n{incorrect_query}"
            }
        ]
        
        print(f"Correcting SQL: {incorrect_query[:50]}...")
        response = self._call_groq_api(messages)
        if 'choices' in response and len(response['choices']) > 0:
            corrected_query = self._extract_sql_from_text(response['choices'][0]['message']['content'])
            return corrected_query
        else:
            print(f"Error in correct_sql response: {response}")
            return incorrect_query
    
    def execute_sql(self, query):
        """
        Execute a SQL query on the database.
        
        :param query: SQL query to execute.
        :return: Dictionary containing query results.
        """
        conn = None
        try:
            conn = psycopg2.connect(
                dbname=self.db_config['dbname'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', '5432')
            )
            cursor = conn.cursor()
            cursor.execute(query)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                results_list = [dict(zip(columns, row)) for row in results]
                cursor.close()
                conn.close()
                return {
                    "success": True,
                    "results": results_list,
                    "row_count": len(results_list)
                }
            else:
                cursor.close()
                conn.close()
                return {
                    "success": True,
                    "message": "Query executed successfully. No data returned."
                }
        except Exception as e:
            if conn:
                conn.close()
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_nl_to_sql_batch(self, data):
        """
        Process a batch of natural language queries to generate SQL queries.
        
        :param data: List of dictionaries with NL queries.
        :return: List of dictionaries with the original NL query and generated SQL.
        """
        results = []
        print(f"Processing {len(data)} NL queries...")
        for i, item in enumerate(data):
            nl_query = item.get("NL", "")
            if not nl_query:
                results.append({"NL": nl_query, "Query": ""})
                continue
            print(f"Processing NL query {i+1}/{len(data)}")
            sql_query = self.generate_sql(nl_query)
            results.append({"NL": nl_query, "Query": sql_query})
            if (i+1) % 5 == 0 or i == len(data) - 1:
                with open('intermediate_sql_generation_task.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved intermediate results: {i+1}/{len(data)} queries processed")
            time.sleep(0.5)
        return results
    
    def process_sql_correction_batch(self, data):
        """
        Process a batch of incorrect SQL queries for correction.
        
        :param data: List of dictionaries with incorrect SQL queries.
        :return: List of dictionaries with the original and corrected SQL queries.
        """
        results = []
        print(f"Processing {len(data)} SQL corrections...")
        for i, item in enumerate(data):
            incorrect_query = item.get("IncorrectQuery", "")
            if not incorrect_query:
                results.append({"IncorrectQuery": incorrect_query, "CorrectQuery": ""})
                continue
            print(f"Processing SQL correction {i+1}/{len(data)}")
            corrected_query = self.correct_sql(incorrect_query)
            results.append({"IncorrectQuery": incorrect_query, "CorrectQuery": corrected_query})
            if (i+1) % 5 == 0 or i == len(data) - 1:
                with open('intermediate_sql_correction_task.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved intermediate results: {i+1}/{len(data)} corrections processed")
            time.sleep(0.5)
        return results

def main():
    db_config = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "admin",
        "host": "localhost",
        "port": "5432"
    }
    # Replace with your valid Groq API key
    api_key = "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"
    
    print("Initializing NLSQL Agent...")
    agent = NLSQLAgent(db_config, api_key, model="llama3-70b-8192")
    
    if not agent.schema:
        print("WARNING: Database schema is empty. Check your database connection.")
    else:
        print(f"Successfully loaded schema for {len(agent.schema)} tables.")
    
    input_file_path_1 = 'sample_submission_generate_task.json'
    input_file_path_2 = 'sample_submission_query_correction_task.json'
    
    specific_path_1 = 'F:/Education/COLLEGE/PROGRAMING/Python/Codes/Text-to-Sql/sample_submission_generate_task.json'
    specific_path_2 = 'F:/Education/COLLEGE/PROGRAMING/Python/Codes/Text-to-Sql/sample_submission_query_correction_task.json'
    
    if not os.path.exists(input_file_path_1) and os.path.exists(specific_path_1):
        input_file_path_1 = specific_path_1
    
    if not os.path.exists(input_file_path_2) and os.path.exists(specific_path_2):
        input_file_path_2 = specific_path_2
    
    try:
        print(f"Loading NL to SQL data from {input_file_path_1}")
        with open(input_file_path_1, 'r') as file:
            nl_sql_data = json.load(file)
        print(f"Loaded {len(nl_sql_data)} NL queries")
    except Exception as e:
        print(f"Error loading NL to SQL data: {e}")
        nl_sql_data = []
    
    try:
        print(f"Loading SQL correction data from {input_file_path_2}")
        with open(input_file_path_2, 'r') as file:
            sql_correction_data = json.load(file)
        print(f"Loaded {len(sql_correction_data)} SQL corrections")
    except Exception as e:
        print(f"Error loading SQL correction data: {e}")
        sql_correction_data = []
    
    if nl_sql_data:
        print("Starting NL to SQL generation...")
        start = time.time()
        sql_statements = agent.process_nl_to_sql_batch(nl_sql_data)
        generate_sqls_time = time.time() - start
        print(f"NL to SQL generation completed in {generate_sqls_time:.2f} seconds")
        with open('output_sql_generation_task.json', 'w') as f:
            json.dump(sql_statements, f, indent=2)
        print("Saved SQL generation results to output_sql_generation_task.json")
    else:
        generate_sqls_time = 0
        sql_statements = []
        print("Skipping NL to SQL generation due to empty dataset")
    
    if sql_correction_data:
        print("Starting SQL correction...")
        start = time.time()
        corrected_sqls = agent.process_sql_correction_batch(sql_correction_data)
        correct_sqls_time = time.time() - start
        print(f"SQL correction completed in {correct_sqls_time:.2f} seconds")
        with open('output_sql_correction_task.json', 'w') as f:
            json.dump(corrected_sqls, f, indent=2)
        print("Saved SQL correction results to output_sql_correction_task.json")
    else:
        correct_sqls_time = 0
        corrected_sqls = []
        print("Skipping SQL correction due to empty dataset")
    
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Time taken to generate SQLs: {generate_sqls_time:.2f} seconds")
    print(f"Time taken to correct SQLs: {correct_sqls_time:.2f} seconds")
    print(f"Total tokens used: {total_tokens}")
    print(f"NL queries processed: {len(sql_statements)}")
    print(f"SQL corrections processed: {len(corrected_sqls)}")
    
    return generate_sqls_time, correct_sqls_time, total_tokens

if __name__ == "__main__":
    main()