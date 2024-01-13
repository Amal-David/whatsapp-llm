import pandas as pd
from collections import defaultdict
import re
import argparse
import datetime
import json

# New function to format data for personal LLM fine-tuning
def format_for_personal_llm_fine_tuning(df, your_name, system_prompt, context_length=3):
    """
    Function to format chat data for fine-tuning a personal LLM. Includes context in each entry.
    """
    formatted_data = []
    conversation_history = []

    for index, row in df.iterrows():
        conversation_history.append((row['Author'], row['Message']))

        # Check if the current message is from 'your_name'
        if row['Author'] == your_name:
            # Accumulate the context messages
            context_messages = ' '.join([f"{author}: {msg}" for author, msg in conversation_history[-(context_length+1):-1]])
            user_response = row['Message']

            # Format the entry
            formatted_entry = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{context_messages} [/INST] {user_response} </s>'
            formatted_data.append({"text": formatted_entry})

            # Reset the conversation history if needed
            conversation_history = []

    return formatted_data


def remove_sensitive_info(text):
    """
    Function to remove sensitive information, specific message types, and URLs from the text.
    """
    # Regex pattern to identify and remove credit card numbers
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[REDACTED CREDIT CARD]', text)

    # Regex pattern to remove basic password patterns
    text = re.sub(r'(?i)\b(password|passwd|pass|pwd)[ :]+[^\s]+\b', '[REDACTED PASSWORD]', text)

    # Regex pattern to remove OTPs (assumed to be 4-6 digits)
    text = re.sub(r'\b\d{4,6}\b', '[REDACTED OTP]', text)

    # Regex pattern to remove PINs (assumed to be 4 digits)
    text = re.sub(r'\b\d{4}\b', '[REDACTED PIN]', text)

    # Remove specific WhatsApp message types and encryption notice
    text = re.sub(r'image omitted', '[REDACTED IMAGE]', text)
    text = re.sub(r'Contact card omitted', '[REDACTED CONTACT CARD]', text)
    text = re.sub(r'video omitted', '[REDACTED VIDEO]', text)
    text = re.sub(r'Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.', '[REDACTED ENCRYPTION NOTICE]', text)
    text = re.sub(r'Waiting for this message. This may take a while.', '[REDACTED WAITING MESSAGE]', text)

    # Regex to remove URLs (both http and https)
    text = re.sub(r'https?://\S+', '[REDACTED URL]', text)

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

    # Create a new DataFrame for fine-tuning
    df_finetune = pd.DataFrame(columns=['Author', 'Message'])

    with open(filepath, 'r', encoding='utf-8') as fp:
        for line in fp:
            # Remove non-ASCII characters except Tamil
            line = re.sub(tamil_regex, '', line)
            # Remove sensitive information
            line = remove_sensitive_info(line)

            match = re.search(pattern, line)
            if match:
                date, time, author, message = match.groups()
                df_finetune = df_finetune.append({'Author': author, 'Message': message}, ignore_index=True)

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

    # Call the format_for_personal_llm_fine_tuning function
    your_name = "Amal David"  # Replace with your name
    system_prompt = "Mimic the conversational style of the user, considering the context of the conversation."  # Replace with your system prompt
    formatted_data = format_for_personal_llm_fine_tuning(df_finetune, your_name, system_prompt)

    return df, formatted_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process your WhatsApp chat data.')
    parser.add_argument('path', type=str, help='Path to file')
    parser.add_argument('prompter', type=str, help='Name of Prompter')
    parser.add_argument('responder', type=str, help='Name of Responder')

    args = parser.parse_args()
    path = args.path
    prompter = args.prompter
    responder = args.responder

    # Convert and parse chat data
    parsed_data, finetune_data = converter_with_debug(path, prompter, responder)
    # print(finetune_data)

    # Save the original parsed data
    parsed_data.to_csv(f'output_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv')
    # Save the formatted data in JSONL format (optional step, depending on your requirements)
    with open('formatted_data.jsonl', 'w') as f:
        for item in finetune_data:
            f.write(json.dumps(item) + '\n')
