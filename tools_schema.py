EMAIL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Sends an email message to a specified recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "The email address of the receiver."},
                "subject": {"type": "string", "description": "The subject of the email."},
                "body": {"type": "string", "description": "The content of the email message."}
            },
            "required": ["recipient", "subject", "body"]
        }
    }
}