from langchain_cohere import ChatCohere
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
# from langchain.tools import tool
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant. If there is url in user query use the crawl_url tool with url as argument.'
                   'However, if there is no url do not do anything, do not apologize simply return the user query'
                   'Example: Crawl www.seinfeld.com. -> crawl_url(www.seinfel.com)'
                   'Example: What is the capital of Taiwan -> What is the capital of Taiwan'
                   ''),
        ('human', "{input}"),
        MessagesPlaceholder("agent_scratchpad")

    ]
)

@tool
def crawl_url(input: int)->int:
    """Crawl the url"""
    # with open("some_file.txt", "w") as file:
    #     file.write(f"{input+2}")
    # return input+2
    return "crawled"

tools = [crawl_url]

llm = ChatCohere()
# llm = llm.bind_tools(tools)

agent=create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result=agent_executor.invoke(
    {
        "input": "?"
    }
)

print(result)

# query = "What is the value of magic_function(2)?"
# out = llm.invoke(query)
# print(out)
