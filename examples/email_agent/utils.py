import os
import asyncio
import boto3
import dotenv

import xml.etree.ElementTree as ET
from botocore.exceptions import ClientError
from aiobotocore.session import get_session

dotenv.load_dotenv()
runtime = boto3.client(
    "bedrock-agent-runtime", 
    region_name=os.environ.get("BEDROCK_REGION")
)
ses_client = boto3.client(
    'ses',
    region_name=os.environ.get("SES_REGION")
)


async def get_user_input(
    placeholder: str
) -> str:
    return input(placeholder)


async def send_email(
    sender_email: str,
    recipient_email: str,
    subject: str,
    body: str,
):
    session = get_session()
    async with session.create_client('ses', os.environ.get("SES_REGION")) as ses_client:
        try:
            response = await ses_client.send_email(
                Source=sender_email,
                Destination={
                    'ToAddresses': [recipient_email]
                },
                Message={
                    'Subject': {
                        'Data': subject
                    },
                    'Body': {
                        'Text': {
                            'Data': body
                        }
                    }
                }
            )
            
            if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                return ("Email sent successfully!")
            else:
                return (f"Failed to send email. Error: {response}")
        except ClientError as e:
            return ("Sending Email error:", e.response['Error']['Message'])


async def retrieve_information(query: str):
    kwargs = {
        "knowledgeBaseId": "VSR83TL8CR",
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": 25,
                "overrideSearchType": "HYBRID"
            }
        },
        "retrievalQuery": {
            "text": query
        }
    }
    
    # Run boto3 call in a thread pool since it's blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: runtime.retrieve(**kwargs))
    return build_context_kb_prompt(result)


def build_context_kb_prompt(
    retrieved_json_file, 
    min_relevant_percentage: float = 0.3,
):
    if not retrieved_json_file:
        return ""
    
    documents = ET.Element("documents")
    
    if retrieved_json_file["ResponseMetadata"]["HTTPStatusCode"] != 200:
        documents.text = "Error in getting data source from knowledge base. No context is provided"
    else:
        body = retrieved_json_file["retrievalResults"]
        for i, context_block in enumerate(body):
            if context_block["score"] < min_relevant_percentage:
                break
            document = ET.SubElement(documents, "document", {"index": str(i + 1)})
            source = ET.SubElement(document, "source")
            content = ET.SubElement(document, "document_content")
            source.text = iterate_through_location(context_block["location"])
            content.text = context_block["content"]["text"]
    
    return ET.tostring(documents, encoding="unicode", method="xml")


def iterate_through_location(location: dict):
    # Optimize by stopping early if uri or url is found
    for loc_data in location.values():
        if isinstance(loc_data, dict):
            uri = loc_data.get("uri")
            if uri:
                return uri
            url = loc_data.get("url")
            if url:
                return url
    return None