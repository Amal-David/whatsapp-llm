# whatsapp-llm

This project is a WhatsApp chat parser. It processes your WhatsApp chat data, redacts sensitive information, and converts it into a CSV file and jsonl. This jsonl file can be used to directly finetune any OSS model for instruction following tasks.

Sensitive information such as phone numbers, credit card numbers, OTP are redacted using a predefined regex pattern. Non-ASCII characters, except those in the Tamil language, are also removed.

To run the script, use the following command:
```python
python parser.py <path_to_file> <prompter> <responder> <your_name>
```
Where:
- `<path_to_file>` is the path to your WhatsApp chat file.
- `<prompter>` is the name of the person who initiates the conversation.
- `<responder>` is the name of the person who responds to the prompter.
- `<your_name>` is a name to be added which can be either prompter or responder.

The script will output a CSV file named 'output_<current_date_and_time>.csv' and jsonl file named formatted_data.jsonl. The CSV file will have two columns: 'prompt' and 'completion'. Each row represents a conversation pair, with 'prompt' being the message from the prompter and 'completion' being the response from the responder. The jsonl file will be a dataset of your personal conversation.

This project is inspired by [chat_scrape](https://github.com/afiqhatta/chat_scrape).
