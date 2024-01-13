import pandas as pd
from collections import defaultdict
import re
import argparse
import datetime


# New function to format data for personal LLM fine-tuning
def format_for_personal_llm_fine_tuning(df, your_name, system_prompt, context_length=3):
    """
    Function to format chat data for fine-tuning a personal LLM. Includes context in each entry.
    """
    formatted_data = []
    conversation_history = []

    for index, row in df.iterrows():
        conversation_history.append((row['Author'], row['Message']))

        if row['Author'] == your_name:
            context_messages = ' '.join([msg for author, msg in conversation_history[-(context_length+1):-1]])
            user_response = row['Message']
            formatted_entry = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{context_messages} [/INST] {user_response} </s>'
            formatted_data.append({"text": formatted_entry})

            # Reset the conversation history if needed
            conversation_history = []

    return formatted_data

def remove_sensitive_info(text):
    """
    Function to remove sensitive information like credit card numbers, passwords, OTPs, and PINs from the text.
    """
    # Regex pattern to identify and remove credit card numbers
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED CREDIT CARD]', text)

    # Regex pattern to remove basic password patterns
    text = re.sub(r'(?i)\b(password|passwd|pass)[ :]+[^\s]+\b', '[REDACTED PASSWORD]', text)

    # Regex pattern to remove OTPs (assumed to be 4-6 digits)
    text = re.sub(r'\b\d{4,6}\b', '[REDACTED OTP]', text)

    # Regex pattern to remove PINs (assumed to be 4 digits)
    text = re.sub(r'\b\d{4}\b', '[REDACTED PIN]', text)

    return text

def converter_with_debug(filepath: str, prompter: str, responder: str) -> pd.DataFrame:
    """
    Converter function to parse chat data, remove sensitive information,
    and create a DataFrame. It pairs one prompt message with one response message.
    """
    # Regex pattern to match the date, time, author, and message
    pattern = r"\[(\d{1,2}/\d{1,2}/\d{2}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APM]{2})\]\s*([^:]+):\s*(.+)"

    # Regex to retain Tamil characters and remove other non-ASCII characters
    tamil_regex = r'[^\x00-\x7F\u0B80-\u0BFF]+'

    result_dict, count = defaultdict(dict), 0
    last_prompt, last_responder = None, None

    with open(filepath, 'r', encoding='utf-8') as fp:
        for line in fp:
            # Remove non-ASCII characters except Tamil language
            line = re.sub(tamil_regex, '', line)
            # Remove sensitive information
            line = remove_sensitive_info(line)

            match = re.search(pattern, line)
            if match:
                date, time, author, message = match.groups()

                if author == prompter:
                    if last_responder is None:  # Waiting for a response
                        last_prompt = message
                    else:
                        count += 1
                        result_dict[count] = {'prompt': last_prompt, 'completion': last_responder}
                        last_prompt, last_responder = message, None

                elif author == responder and last_prompt is not None:
                    last_responder = message

    # Add the last pair if it exists
    if last_prompt and last_responder:
        count += 1
        result_dict[count] = {'prompt': last_prompt, 'completion': last_responder}

    print("Number of entries in result_dict:", len(result_dict))

    df = pd.DataFrame.from_dict(result_dict, orient='index')

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process your WhatsApp chat data.')
    parser.add_argument('path', type=str, help='Path to file')
    parser.add_argument('prompter', type=str, help='Name of Prompter')
    parser.add_argument('responder', type=str, help='Name of Responder')

    args = parser.parse_args()
    path = args.path
    prompter = args.prompter
    responder = args.responder

    converter_with_debug(path, prompter, responder).to_csv(f'output_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv')
