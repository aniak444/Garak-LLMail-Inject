import logging

class MockEmailAgent:
    """
    Simulates an email tool environment for testing purposes.
    """
    def __init__(self):
        self.history = []
        logging.basicConfig(level=logging.INFO)

    def send_email(self, recipient: str, subject: str, body: str):
        """Simulates sending an email by capturing the call parameters."""
        action = {
            "tool": "send_email",
            "recipient": recipient,
            "subject": subject,
            "body": body
        }
        self.history.append(action)
        logging.info(f"MOCK TOOL TRIGGERED: Sent email to {recipient}")
        return "Success: Email sent."

    def get_triggered_actions(self):
        """Retrieves a list of all tool calls recorded during the test session."""
        return self.history