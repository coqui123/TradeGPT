"""
JSON Utilities
Contains functions for handling JSON data
"""
import json
import logging
import re
from decimal import Decimal
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder for Decimal objects"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, bool):  # Add handling for boolean values
            return bool(obj)
        return super(DecimalEncoder, self).default(obj)

def format_json_result(result: str) -> Union[Dict, List[Dict], None]:
    """Format and validate JSON result from LLM response."""
    try:
        # Remove leading/trailing whitespace
        result = result.strip()
        
        # Find JSON content between the last set of triple backticks if present
        if "```json" in result and "```" in result:
            start = result.rindex("```json") + 7
            end = result.rindex("```")
            json_content = result[start:end].strip()
        # If no backticks, try to find JSON between curly braces
        elif "{" in result and "}" in result:
            start = result.find("{")
            end = result.rfind("}") + 1
            json_content = result[start:end].strip()
        else:
            json_content = result
        
        # Remove comments (both // and /* */ style)
        json_content = re.sub(r'//.*?$', '', json_content, flags=re.MULTILINE)  # Remove // comments
        json_content = re.sub(r'/\*.*?\*/', '', json_content, flags=re.DOTALL)  # Remove /* */ comments
            
        # Pre-process mathematical expressions
        def evaluate_expression(match):
            try:
                expression = match.group(1)
                # Handle trailing decimal points
                expression = expression.rstrip('.')
                # Evaluate the expression
                result = eval(expression)
                # Format the result to 8 decimal places and remove trailing zeros
                return f"{float(result):.8f}".rstrip('0').rstrip('.')
            except:
                return match.group(0)

        # Handle mathematical expressions in the JSON
        # Handle basic arithmetic (including decimal numbers)
        json_content = re.sub(r'([-+]?\d*\.?\d+\s*[-+*/]\s*[-+]?\d*\.?\d+(?:\s*[-+*/]\s*[-+]?\d*\.?\d+)*)', 
                            evaluate_expression, 
                            json_content)
        
        # Handle specific patterns like "x * 0.95" or similar
        json_content = re.sub(r'(\d+\.?\d*)\s*\*\s*0\.95', 
                            lambda m: str(float(m.group(1)) * 0.95), 
                            json_content)

        # Handle addition with ATR or other variables
        json_content = re.sub(r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)', 
                            lambda m: str(float(m.group(1)) + float(m.group(2)) * float(m.group(3))), 
                            json_content)

        # Clean up any remaining mathematical operators
        json_content = re.sub(r'\s*[-+*/]\s*', '', json_content)
        
        # Parse the JSON content
        parsed_json = json.loads(json_content)
        
        # Convert all numeric values to proper format
        def format_numeric_values(obj):
            if isinstance(obj, dict):
                return {k: format_numeric_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [format_numeric_values(v) for v in obj]
            elif isinstance(obj, (int, float)):
                return float(f"{obj:.8f}".rstrip('0').rstrip('.'))
            return obj

        parsed_json = format_numeric_values(parsed_json)
        
        # Handle both single object and array responses
        if isinstance(parsed_json, dict):
            return [parsed_json]
        return parsed_json
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from result: {result}")
        logger.error(f"JSON parse error: {e}")
        # Try to clean up the JSON string more aggressively
        try:
            # Remove all comments (both // and /* */ style)
            json_content = re.sub(r'//.*?$', '', json_content, flags=re.MULTILINE)
            json_content = re.sub(r'/\*.*?\*/', '', json_content, flags=re.DOTALL)
            
            # Remove all whitespace between values and operators
            json_content = re.sub(r'\s*([-+*/])\s*', r'\1', json_content)
            
            # Evaluate any remaining mathematical expressions
            json_content = re.sub(r'([-+]?\d*\.?\d+[-+*/]\d*\.?\d+)', 
                                lambda m: str(eval(m.group(1))), 
                                json_content)
            
            # Try to fix common JSON syntax errors
            # Replace single quotes with double quotes for keys and string values
            json_content = re.sub(r'\'([^\']+)\'', r'"\1"', json_content)
            
            # Remove trailing commas in objects and arrays
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            
            parsed_json = json.loads(json_content)
            if isinstance(parsed_json, dict):
                return [parsed_json]
            return parsed_json
        except Exception as e2:
            logger.error(f"Second attempt at JSON parsing failed: {e2}")
            
            # Last resort: try to extract a valid JSON object using regex
            try:
                # Find patterns that look like valid JSON objects
                pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
                matches = re.findall(pattern, json_content)
                if matches:
                    for potential_json in matches:
                        try:
                            parsed = json.loads(potential_json)
                            return [parsed]
                        except:
                            continue
            except:
                pass
                
            return None
    except Exception as e:
        logger.error(f"Unexpected error in format_json_result: {str(e)}")
        return None

def format_decimal(value):
    """Format decimal values properly"""
    if value is None:
        return None
    try:
        # Convert to Decimal for precise handling
        decimal_value = Decimal(str(value))
        # Remove trailing zeros after decimal point
        normalized = decimal_value.normalize()
        # Convert to string and handle scientific notation
        if 'E' in str(normalized):
            # Handle very small or large numbers
            return f"{decimal_value:.8f}".rstrip('0').rstrip('.')
        return str(normalized)
    except:
        return str(value) 