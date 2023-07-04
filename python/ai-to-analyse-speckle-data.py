import os
import json
from datetime import datetime
import requests
import re
from getpass import getpass
import glob
from PyPDF2 import PdfReader

from flask import Flask, request
from flask_executor import Executor

import openai

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from langchain.text_splitter import NLTKTextSplitter

import weaviate

from specklepy.api.client import SpeckleClient

### ENVIROMENT ###

# should match the webhook setup in speckle admin
flask_url = '0.0.0.0'
flask_port = 5000
flask_path = '/webhook'

weaviate_url = 'http://localhost:8080/'

speckle_url = 'http://localhost/'

# probably not a very safe solution...
# get api key from https://platform.openai.com/account/api-keys
os.environ["OPENAI_API_KEY"] = "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"
# get access token for "ai user"
os.environ["SPECKLE_TOKEN"] = "19ba69db13a9c60f3992627a91d2747bc68e5d0cc7"

# for legacy conversation history stuffing
openai.api_key = os.environ["OPENAI_API_KEY"]

#for weaviate
api_key = os.environ["OPENAI_API_KEY"]

speckle_client = SpeckleClient(host="localhost", use_ssl=False)
speckle_client.authenticate_with_token(os.environ["SPECKLE_TOKEN"])

### WEAVIATE ###
client = weaviate.Client(
    url=weaviate_url,
    additional_headers={
        "X-OpenAI-API-Key": api_key
    }
)

def weaviate_document(text, filename):
    data_object = {
        "text": text,
        "filename": filename
        }
    weaviate_id = client.data_object.create(data_object, class_name="Documents")
    return weaviate_id

### SPECKLE ###
def extract_comment_data(payload):
    stream_id = payload['streamId']
    
    activity_message = payload['activityMessage']
    thread_id = activity_message.split(': ')[1].split(' ')[0]
    
    text_data = payload['event']['data']['input']['text']['content'][0]['content'][0]['text']
    # split text by spaces and pick first word
    object_id = text_data.split(' ')[0]
    # rest of the text is user question
    user_question = ' '.join(text_data.split(' ')[1:]) 
    return stream_id, object_id, user_question, thread_id

def extract_reply_data(payload):
    stream_id = payload['streamId']
    user_id = payload['userId']

    activity_message = payload['activityMessage']
    comment_id = activity_message.split('#')[-1].split(' ')[0] if '#' in activity_message else None

    if 'parentComment' in payload['event']['data']['input']:
        thread_id = payload['event']['data']['input']['parentComment']
        text_data = payload['event']['data']['input']['text']['content'][0]['content'][0]['text']
    else:
        thread_id = payload['event']['data']['input']['threadId']
        text_data = payload['event']['data']['input']['content']['doc']['content'][0]['content'][0]['text']

    user_question = text_data.strip()

    return stream_id, user_id, user_question, thread_id, comment_id

# important note: only data under "parameters" will be returned. this works well for revit objects but has not been tested on other types
def extract_data_to_json(speckle_data, object_id):
    parameters_data = speckle_data['stream']['object']['data']['parameters']
    result_dict = {}

    for key in parameters_data.keys():
        # check if dict
        if isinstance(parameters_data[key], dict):
            name = parameters_data[key]['name']
            units = parameters_data[key]['units']
            if units is not None:
                key_name = f"{name} ({units})"
            else:
                key_name = name
            result_dict[key_name] = parameters_data[key]['value']
        else:
            print(f"Unexpected data format for key {key}: {parameters_data[key]}")

    # save as json
    with open(f'webhook/{object_id}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    # convert dict to json string
    result_json = json.dumps(result_dict)

    return result_json

def extract_and_transform_thread(raw_thread: Dict[str, Any], thread_id: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Extracts and transforms a thread from raw thread data.

    Args:
        raw_thread (dict): The raw thread data.
        thread_id (str): The ID of the thread to extract.

    Returns:
        tuple: A tuple containing a list of conversation strings and the object ID, or (None, None) if the thread is not found.
    """
    threads = raw_thread['project']['commentThreads']['items']
    for thread in threads:
        if thread['id'] == thread_id:
            # Extract object_id and question
            object_id, question = thread['rawText'].split(' ', 1)
            conversation = [f"Question: {question}"]
            # Sort replies by createdAt
            replies = sorted(thread['replies']['items'], key=lambda x: x['createdAt'])
            for reply in replies:
                conversation.append(f"Answer: {reply['rawText']}" if reply['authorId'] == '3952b2a678' else f"Question: {reply['rawText']}")
            return conversation, object_id
    return None, None


def speckle_graphql_query(query, variables=None):
    url = f"{speckle_url}graphql"
    payload = {"query": query, "variables": variables}
    token = os.environ['SPECKLE_TOKEN']
    headers = {"Authorization": token, "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

def speckle_get_data(stream_id, object_id):
    query = """
        query Query($streamId: String!, $objectId: String!) {
            stream(id: $streamId) {
                object(id: $objectId) {
                    id
                    data
                }
            }
        }
    """
    variables = {
        "streamId": stream_id,
        "objectId": object_id
    }
    data = speckle_graphql_query(query, variables)
    if data and 'data' in data:
        return data['data']
    else:
        return None
    
def speckle_reply(thread_id, reply_text):
    mutation = """
        mutation Mutation($input: CreateCommentReplyInput!) {
            commentMutations {
                reply(input: $input) {
                    id
                }
            }
        }
    """
    input_content = {
        "doc": {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": reply_text
                        }
                    ]
                }
            ]
        }
    }
    variables = {
        "input": {
            "content": input_content,
            "threadId": thread_id
        }
    }
    data = speckle_graphql_query(mutation, variables)
    if data and 'data' in data:
        return data['data']
    else:
        return None

def speckle_thread(stream_id):
    query = """
        query Query($projectId: String!) {
          project(id: $projectId) {
            commentThreads {
              items {
                id
                rawText
                replies {
                  items {
                    id
                    authorId
                    createdAt
                    rawText
                  }
                }
              }
            }
          }
        }
    """
    variables = {
        "projectId": stream_id,
    }
    data = speckle_graphql_query(query, variables)
    if data and 'data' in data:
        return data['data']
    else:
        return None

def extract_text_from_pdf(pdf_path):
    # using PyPDF2. "unstructured" is more powerful but also much more complicated.
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_attachment(stream_id, attachment_id, thread_specific_filename, thread_id, folder="attachments"):
    print(f"stream_id: {stream_id}")
    print(f"attachment_id: {attachment_id}")
    print(f"filename: {thread_specific_filename}")
    url = f"{speckle_url}api/stream/{stream_id}/blob/{attachment_id}"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, thread_specific_filename), 'wb') as f:
            f.write(response.content)
        print(f"File {thread_specific_filename} downloaded successfully")

        # extract and import data
        text = extract_text_from_pdf(os.path.join(folder, thread_specific_filename))

        # use nltk_text_splitter to split text
        nltk_text_splitter = NLTKTextSplitter(chunk_size=1000)
        chunks = nltk_text_splitter.split_text(text)

        # create chunks in weaviate
        for chunk in chunks:
            data_object = {
                "text": chunk,
                "filename": thread_specific_filename
            }
            weaviate_id = client.data_object.create(data_object, class_name="Documents")
            print(f"Weaviate ID for the created chunk: {weaviate_id}")
        
        print(f"File {thread_specific_filename} imported successfully")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def attachment(comment_id, thread_id, stream_id):
    query = """
        query Query($commentId: String!, $streamId: String!) {
            comment(id: $commentId, streamId: $streamId) {
                id
                text {
                    attachments {
                        id
                        fileName
                        fileHash
                        fileType
                        fileSize
                    }
                }
            }
        }
    """
    variables = {
        "commentId": comment_id,
        "streamId": stream_id,
    }
    data = speckle_graphql_query(query, variables)
    
    if data and 'data' in data:
        attachments = data['data']['comment']['text']['attachments']
        for attachment in attachments:
            if attachment['fileType'].lower() == 'pdf':
                thread_specific_filename = f"{thread_id}_{attachment['fileName']}"
                get_attachment(stream_id, attachment['id'], thread_specific_filename, thread_id)
                return thread_specific_filename
            else:
                print(f"Skipped non-pdf file: {attachment['fileName']}")
    else:
        print("Failed to fetch comment attachments")

### OPENAI ###

# legacy gpt agent for conversation history stuffing. works well but there might be a neater solution built into langchain
# def gpt(system_prompt, user_prompt, model="gpt-3.5-turbo", n=1, max_tokens=1000):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         n=n,
#         max_tokens=max_tokens,
#         temperature=0,
#     )
#     return response.choices[0].message['content'].strip()

# legacy gpt prompt for conversation history stuffing. works well but there might be a neater solution built into langchain
# def gpt_standalone(conversation):
#     system_prompt = f"""
# Your task is to analyze the provided conversation history and formulate a standalone question so that it makes sense to the receiver. Be concise, no need for pleasantries.

# Conversation history:
# {conversation}
# """
#     user_prompt = f"""
# Your forwarded question:
# """
#     gpt_response = gpt(system_prompt, user_prompt)
#     print(f"gpt_response: {gpt_response}")
#     return gpt_response


### LANGCHAIN ###

from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.docstore.document import Document

# two options for llm, gpt3 (davinci) and gpt-3.5. the latter seems to be quicker and smarter but not as good following instructions. i had more success with davinci. gpt-3.5 might need custom parser to handle when it goes off track.
llm = OpenAI(temperature=0, model="text-davinci-003", openai_api_key=os.environ["OPENAI_API_KEY"])
chat_llm = OpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.environ["OPENAI_API_KEY"])

def gpt_standalone(text):
    prompt_template = """Your task is to analyze the provided conversation history and formulate a standalone question so that it makes sense to the receiver. Be concise, no need for pleasantries.

Conversation history:

{text}


Your forwarded question:"""

    STUFF_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=STUFF_PROMPT)

    # converting as langchain summarize needs doc input, there may be a neater solution
    doc = [Document(page_content=text)]

    gpt_response = chain.run(doc)
    print(f"gpt_response: {gpt_response}")
    return gpt_response


# custom prompt template, changes made here greatly affect the output
template = """You are a helpful assistant that follows instructions extremely well. Answer the question regarding a certain object in a BIM model as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!

Question: {input}

{agent_scratchpad}"""

# default langchain stuff from here on:

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    # Added data_json as a variable for the class
    data_json: str = ''

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # Include data_json in the formatted string
        kwargs["data_json"] = self.data_json
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

def create_prompt(tools):
    return CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

### EXECUTORS ###

# new comment:
def comment(data, stream_id, object_id, user_question, thread_id):
    #save payload as file (for troubleshooting)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'webhook/{timestamp}_{object_id}_{thread_id}_comment.json', 'w') as f:
        json.dump(data, f)

    print(f"stream_id: {stream_id}")
    print(f"object_id: {object_id}")
    print(f"user_question: {user_question}")
    print(f"thread_id: {thread_id}")
    
    # define tool for langchain agent to fetch speckle data. defined within executor due to langchain custom variable limitations/challenges.
    def get_data(input):
        # get all data for object
        speckle_data = speckle_get_data(stream_id, object_id)
        # extract parameter data
        data_json = extract_data_to_json(speckle_data, object_id)
        # pretty-print
        data_formatted = json.dumps(data_json, indent=4)
        # providing context with data improves gpt response
        description = f"All available parameter data has been provided below related to {input}, choose suitable parameter value(s) matching the question. All units are metric.\n"
        description_and_data = description + data_formatted
        return description_and_data

    get_data_tool = Tool(
        name="DataSearch",
        func=get_data,
        description=f"Useful when additional data is needed. All data relevant to the question data will be provided. After 'Action input:', you must provide a single search string within ticks in the following format: 'search_term'"
    )

    tools = [get_data_tool]

    # the following is only for when there is attachment in current comment. defined within executor due to langchain custom variable limitations/challenges.
    comment_id = thread_id
    filename = None
    filename = attachment(comment_id, thread_id, stream_id)
    print(f"filename: {filename}")

    def weaviate_neartext(keyword, filename=filename, limit=2):
        near_text = {"concepts": keyword}
        query = (
            client.query
            .get("Documents", ["text", "filename"])
            .with_additional(["distance"])
            .with_near_text(near_text)
            .with_limit(limit)
        )
        if filename:
            where_filter = {
                "path": ["filename"],
                "operator": "Equal",
                "valueText": filename
            }
            query = query.with_where(where_filter)
        results = query.do()
        return results

    weaviate_neartext_tool = Tool(
        name = 'DocSearch',
        func = weaviate_neartext,
        description = f"Used for searching in attached document(s). After 'Action input:', you must provide a single search string within ticks in the following format: 'search_term'"
    )

    if filename is not None:
        tools.append(weaviate_neartext_tool)

    tool_names = [tool.name for tool in tools]
    print(f"tool_names: {tool_names}")

    # initiate and run langchain agent
    prompt = create_prompt(tools)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    final_answer = agent_executor.run(user_question)

    # post answer as reply comment
    speckle_reply(thread_id, final_answer)

# new reply
def reply(data, stream_id, user_id, user_question, thread_id, comment_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'webhook/{timestamp}_{thread_id}_reply.json', 'w') as f:
        json.dump(data, f)

    print(f"stream_id: {stream_id}")
    print(f"user_id: {user_id}")
    print(f"user_question: {user_question}")
    print(f"thread_id: {thread_id}")
    print(f"comment_id: {comment_id}")

    # get full comment thread, extract object_id from first comment
    raw_thread = speckle_thread(stream_id)
    conversation, object_id = extract_and_transform_thread(raw_thread, thread_id)

    print(f"conversation: {conversation}")
    print(f"object_id: {object_id}")

    # check if first comment has object_id (<25 chars)
    if len(object_id) >= 25:

        # use openai to compose standalone question from conversation history (legacy code, should probably be replaced with langchain equivalent)
        question = gpt_standalone(conversation)
        
        # define tool for langchain agent to fetch speckle data. defined within executor due to langchain custom variable limitations/challenges.
        def get_data(input):
            # get all data for object
            speckle_data = speckle_get_data(stream_id, object_id)
            # extract parameter data
            data_json = extract_data_to_json(speckle_data, object_id)
            # pretty-print
            data_formatted = json.dumps(data_json, indent=4)
            # providing context with data improves gpt response
            description = f"All available parameter data has been provided below related to {input}, choose suitable parameter value(s) matching the question. All units are metric.\n"
            description_and_data = description + data_formatted
            return description_and_data

        get_data_tool = Tool(
            name="DataSearch",
            func=get_data,
            description=f"Useful when additional data is needed. All data relevant to the question data will be provided. After 'Action input:', you must provide a single search string within ticks in the following format: 'search_term'"
        )

        tools = [get_data_tool]

        # the following is only for when there is attachment in current or previous comments. defined within executor due to langchain custom variable limitations/challenges.
        filenames = []
        attachment_filename = attachment(comment_id, thread_id, stream_id)
        if attachment_filename:
            filenames.append(attachment_filename)
        previous_attachments = [os.path.basename(f) for f in glob.glob(f"attachments/{thread_id}_*")]
        filenames.extend(previous_attachments)
        print(f"filenames: {filenames}")

        def weaviate_neartext(keyword, filenames=filenames, limit=2):
            near_text = {"concepts": keyword}
            query = (
                client.query
                .get("Documents", ["text", "filename"])
                .with_additional(["distance"])
                .with_near_text(near_text)
                .with_limit(limit)
            )
            if filenames:
                # changed where filter to handle multiple pdf's in one comment thread
                where_filter = {
                    "operator": "Or",
                    "operands": [
                        {
                            "path": ["filename"],
                            "operator": "Equal",
                            "valueString": filename
                        }
                        for filename in filenames
                    ]
                }
                query = query.with_where(where_filter)
            results = query.do()
            return results

        weaviate_neartext_tool = Tool(
            name = 'DocSearch',
            func = weaviate_neartext,
            description = f"Used for searching in attached document(s). After 'Action input:', you must provide a single search string within ticks in the following format: 'search_term'"
        )

        if filenames:
            tools.append(weaviate_neartext_tool)

        tool_names = [tool.name for tool in tools]
        print(f"tool_names: {tool_names}")

        # initiate and run langchain agent
        prompt = create_prompt(tools)

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

        final_answer = agent_executor.run(question)

        # post answer as reply comment
        speckle_reply(thread_id, final_answer)

    return 

app = Flask(__name__)
executor = Executor(app)

@app.route(flask_path, methods=['POST'])

# in speckle admin, setup webhook for new comments and replies
def handle_webhook():
    data = request.get_json()
    # identify comment/reply
    event_name = data['payload']['event']['event_name']
    print(f"event_name: {event_name}")

    if event_name == 'comment_created':
        print("new comment")
        # comment must contain object_id as first word (due to current graphql/webhook limitations)
        stream_id, object_id, user_question, thread_id = extract_comment_data(data['payload'])
        # if first word in comment is object_id (>25 chars), run executor. should be replaced with "@ai" or similar when object_id can be fetched via webhook.
        if len(object_id) > 25:
            executor.submit(comment, data, stream_id, object_id, user_question, thread_id)

    elif event_name == 'comment_replied':
        print("new reply")
        stream_id, user_id, user_question, thread_id, comment_id = extract_reply_data(data['payload'])
        # confirm that reply is not ai generated
        if user_id != '3952b2a678':
            executor.submit(reply, data, stream_id, user_id, user_question, thread_id, comment_id)
        else:
            print("ignored: ai reply, or new comment without id")

    return '', 200

if __name__ == '__main__':
    if not os.path.exists('webhook'):
        os.makedirs('webhook')
    app.run(host=flask_url, port=flask_port)