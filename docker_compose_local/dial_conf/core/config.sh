#!/bin/sh
set -e

echo "Debug: Starting the script"

# Set the input and output files
INPUT_FILE="/opt/config/config-template.json"
OUTPUT_FILE="/opt/config/config.json"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file $INPUT_FILE not found."
  exit 1
fi

# Check for the DEPLOY_DIAL_RAG flag
echo "should be dial-rag deployed? ${DEPLOY_DIAL_RAG}"
if [ "${DEPLOY_DIAL_RAG}" = "1" ]; then
  DIAL_RAG_URL="http://dial-rag:5000"
fi

# Function to substitute environment variables
substitute_env_variables() {
  template_file="$1"
  output_file="$2"

  # Clear the output file
  > "$output_file"

  # Initialize a counter for substitutions
  substitution_count=0

  # Read the template file line by line
  while IFS= read -r line || [ -n "$line" ]; do
    # Initialize modified_line with the original line
    modified_line="$line"

    placeholders=$(echo "$line" | grep -o '\${[^}]*}' | sort -u)

    # Debug: Echo all placeholders found in the input file
#    if [ -n "$placeholders" ]; then
#      echo "Found placeholders: $placeholders"
#    fi

    # Substitute environment variables in the line
    for placeholder in $placeholders; do
      # Extract the variable name from the placeholder
      var_name="${placeholder#\${}"
      var_name="${var_name%\}}"

      # Get the value of the environment variable
      env_value=$(eval "echo \$$var_name")

      # Debug: Echo the value that would be substituted
      if [ -n "$env_value" ]; then
        # echo "Substituting: $placeholder with value: $env_value"
        modified_line="${modified_line//$placeholder/$env_value}"
        substitution_count=$((substitution_count + 1))
      else
        # Handle default value syntax
        default_value=$(echo "$line" | sed -E "s/\$\{${var_name}:-([^}]+)\}/\1/g")
        modified_line="${modified_line//$placeholder/$default_value}"
        # If a default value is used, count it as a substitution
        if [ "$modified_line" != "$line" ]; then
          substitution_count=$((substitution_count + 1))
        fi
      fi
    done

    # Write the modified line to the output file
    echo "$modified_line" >> "$output_file"
  done < "$template_file"

  # Echo statistics about substitutions made
  echo "Total substitutions made: $substitution_count"
}

# Call the substitution function
substitute_env_variables "$INPUT_FILE" "$OUTPUT_FILE"

echo "Substitution completed. Output written to $OUTPUT_FILE."