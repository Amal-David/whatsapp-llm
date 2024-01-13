# whatsapp-llm

This project is a WhatsApp chat parser. It processes your WhatsApp chat data and converts it into a CSV file.

To run the script, use the following command:
```python
python parser.py <path_to_file> <prompter> <responder>
```
Where:
- `<path_to_file>` is the path to your WhatsApp chat file.
- `<prompter>` is the name of the person who initiates the conversation.
- `<responder>` is the name of the person who responds to the prompter.

The script will output a CSV file named 'output_<current_date_and_time>.csv'.
