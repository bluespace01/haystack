# import os
# from haystack import Pipeline, PredefinedPipeline

# os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"

# pipeline = Pipeline.from_template(PredefinedPipeline.CHAT_WITH_WEBSITE)
# result = pipeline.run({
#     "fetcher": {"urls": ["https://haystack.deepset.ai/overview/quick-start"]},
#     "prompt": {"query": "Which components do I need for a RAG pipeline?"}}
# )
# print(result["llm"]["replies"][0])


# https://haystack.deepset.ai/integrations/groq
from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

fetcher = LinkContentFetcher()
converter = HTMLToDocument()
prompt_template = """
According to the contents of this website:
{% for document in documents %}
  {{document.content}}
{% endfor %}
Answer the given question: {{query}}
Answer:
"""
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="mixtral-8x7b-32768",
    generation_kwargs = {"max_tokens": 512}
)
pipeline = Pipeline()
pipeline.add_component("fetcher", fetcher)
pipeline.add_component("converter", converter)
pipeline.add_component("prompt", prompt_builder)
pipeline.add_component("llm", llm)

pipeline.connect("fetcher.streams", "converter.sources")
pipeline.connect("converter.documents", "prompt.documents")
pipeline.connect("prompt.prompt", "llm.prompt")

result = pipeline.run({"fetcher": {"urls": ["https://wow.groq.com/why-groq/"]},
              "prompt": {"query": "What is the purpose of Groq?"}})

print(result["llm"]["replies"][0])
