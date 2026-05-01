# for gemini
EMAIL_TOOL_SCHEMA = {
    "name": "send_email",
    "description": "Sends an email message to a specified recipient.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "recipient": {
                "type": "STRING", 
                "description": "The email address of the receiver."
            },
            "subject": {
                "type": "STRING", 
                "description": "The subject of the email."
            },
            "body": {
                "type": "STRING", 
                "description": "The content of the email message."
            }
        },
        "required": ["recipient", "subject", "body"]
    }
}