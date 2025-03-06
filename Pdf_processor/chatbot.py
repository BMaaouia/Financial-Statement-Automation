import aiml

# Create the Kernel (the chatbot)
kernel = aiml.Kernel()

# Use the 'standard' AIML set
kernel.bootstrap(learnFiles="C://Users/Maaouia/Desktop/FinancialStatementAutomation/Pdf_processor/std-startup.xml", commands="load aiml b")

# Set a session ID to keep conversations separate
session_id = "1"

def get_response(user_input):
    # Return the chatbot's response to the user input
    return kernel.respond(user_input, session_id)
