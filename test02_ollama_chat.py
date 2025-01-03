from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.ollama import OllamaChatGenerator

messages = [
    ChatMessage.from_user("What's Natural Language Processing?"),
    ChatMessage.from_system(
        "Natural Language Processing (NLP) is a field of computer science and artificial "
        "intelligence concerned with the interaction between computers and human language"
    ),
    ChatMessage.from_user("How do I get started?"),
]
# client = OllamaChatGenerator(model="orca-mini", timeout=45, url="http://localhost:11434")
client = OllamaChatGenerator(model="llama3.1", timeout=45, url="http://localhost:11434")


response = client.run(messages, generation_kwargs={"temperature": 0.2})

print(response["replies"][0].text)
