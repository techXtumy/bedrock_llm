import asyncio
import sys
import os
import random
import pytz

from termcolor import cprint
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient
from src.bedrock_llm.types.enums import ModelName
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.tools import ToolMetadata, InputSchema, PropertyAttr
from src.bedrock_llm.schema.message import MessageBlock, ToolUseBlock, TextBlock, ToolResultBlock
from src.bedrock_llm.utils.prompt import llama_format

from typing import Literal, List, Optional

model_config = ModelConfig(
    temperature=0.9,
    max_tokens=2048,
    top_p=0.9
)
retry_config = RetryConfig(
    max_retries=3,
    retry_delay=0.5
)


# Function to handle user input asynchronously
async def get_user_input(placeholder: str) -> str:
    return input(placeholder)


async def get_company_info(
    company_name: str, 
    start_year: int, 
    end_year: int, 
    company_type: Literal["corp", "inc", "llc"]
) -> str:
    loop = asyncio.get_event_loop()
    def foo(company_name: str, 
            start_year: int, 
            end_year: int, 
            company_type: Literal["corp", "inc", "llc"]):
        # Predefine company types with related info
        company_data = {
            "corp": {"tax_rate": 0.25, "legal_structure": "Corporation"},
            "inc": {"tax_rate": 0.22, "legal_structure": "Incorporated"},
            "llc": {"tax_rate": 0.20, "legal_structure": "Limited Liability Company"}
        }
        
        # Get specific details for the company type
        tax_rate = company_data[company_type]["tax_rate"]
        legal_structure = company_data[company_type]["legal_structure"]
        
        # Generate basic company details
        founding_year = random.randint(1900, start_year)
        industry = random.choice(
            ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"]
        )
        
        # Prepare company info header
        info = [
            f"Company: {company_name}",
            f"Type: {company_type.upper()} ({legal_structure})",
            f"Industry: {industry}",
            f"Founded: {founding_year}"
        ]
        
        # Generate records for each year
        for year in range(start_year, end_year + 1):
            is_disabled = random.choice([True, False])
            number_of_employees = random.randint(100, 1000)
            annual_revenue = random.randint(1_000_000, 100_000_000)
            
            if is_disabled:
                status = "Currently inactive"
                number_of_employees = int(number_of_employees * random.uniform(0.1, 0.5))
            else:
                status = "Fully operational"
            
            estimated_taxes = annual_revenue * tax_rate
            
            # Add yearly info to the list
            info.append(f"\nYear: {year}")
            info.append(f"Status: {status}")
            info.append(f"Employees: {number_of_employees}")
            info.append(f"Annual Revenue: ${annual_revenue:,}")
            info.append(f"Estimated Taxes: ${estimated_taxes:,.2f}")
        
        # Join list into a single string
        return "\n".join(info)
    
    return await loop.run_in_executor(
        None, foo, company_name, start_year, end_year, company_type)


async def get_imagine_news(
    first_company_name: str,
    start_year: int,
    end_year: int,
    second_company_name: Optional[str] = None,
    third_company_name: Optional[str] = None
) -> str:
    
    if second_company_name is None:
        second_company_name = "Gabbie Carter"
    if third_company_name is None:
        third_company_name = "Dainty Wilder"
    
    system = "You are a news generator. Your job is to generate a imagine world where it is full of drama and twist about economy, finacial, stock market and all its related field. Please provide the story in full detail."
    user = f"Imagine a world where {first_company_name} and {second_company_name} are in a stock-take competition. {third_company_name} is a big investor in the market. Please generate a news for this situation in {start_year} to {end_year}."
    
    llama_client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.LLAMA_3_2_90B,
        retry_config=retry_config
    )
    async for message, stop_reason in await llama_client.generate(
        prompt=llama_format(user_messages=user, system=system),
        config=model_config
    ):
        cprint(message, color="magenta", end="", flush=True)
        if stop_reason:
            break
        
    return message


async def get_trade_advice(
    company_name: str,
    start_year: int,
    end_year: int
) -> str:

    system = "Assuming you are living in another world. In the world, you are a financial advisor. Your job is to generate or provide a short unreal advice for investment decision based on an imanination company name. Please be creative and original."
    user = f"Please provide trade advice for {company_name} in {start_year} to {end_year}."

    llama_client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.LLAMA_3_2_90B,
        retry_config=retry_config
    )
    async for message, stop_reason in await llama_client.generate(
        prompt=llama_format(user_messages=user, system=system),
        config=model_config
    ):
        cprint(message, color="blue", end="", flush=True)
        if stop_reason:
            break

    return message


async def process_tools(
    tools_list: List[ToolUseBlock]
) -> MessageBlock:
    message = MessageBlock(role="user", content=[])
    
    for tool in tools_list:
        if isinstance(tool, TextBlock):
            continue
        
        tool_function = tool.name
        if tool_function:
            try:
                result = await globals()[tool_function](**tool.input)
                is_error = False
            except Exception as e:
                result = str(e)
                is_error = True
        else:
            result = f"Tool {tool.name} not found"
            is_error = True
            
        message.content.append(
            ToolResultBlock(
                type="tool_result",
                tool_use_id=tool.id,
                is_error=is_error,
                content=result
            )
        )
    
    return message


async def titan_agent():
    pass


async def llama_agent():
    pass


async def claude_agent():
    # Initialize the client
    client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.CLAUDE_3_5_HAIKU,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    # Initialize model configuration
    config = ModelConfig(
        temperature=0.5, 
        max_tokens=2048, 
        top_p=0.9, 
        top_k=70
    )
    
    # Initialize params
    chat_history = []
    system_prompt = f"You are a helpful assistant. This is the real time data {datetime.now(tz = pytz.timezone("Asia/Bangkok")).strftime('%Y-%m-%d %H:%M:%S %Z')}"
    message = "Comparing and analyzing the Sigma LLC and Skibidi Toilet stock-joint comanpy from 2016 till now. Which company should I invest stock for"
    chat_history.append(
        MessageBlock(
            role="user",
            content=message
        ).model_dump()
    )
    
    # Define tools metadata
    get_company_info_tool = ToolMetadata(
        name="get_company_info",
        description="Get the current information of a company",
        input_schema=InputSchema(
            type="object",
            properties={
                "company_name": PropertyAttr(
                    type="string",
                    description="The name of the company that you are searching for"
                ),
                "start_year": PropertyAttr(
                    type="integer",
                    description="The start year of the finacial report"
                ),
                "end_year": PropertyAttr(
                    type="integer",
                    description="The end year of the finacial report"
                ),
                "company_type": PropertyAttr(
                    type="string",
                    enum=["corp", "inc", "llc"],
                    description="The type of the company (corp, inc, llc)"
                )
            },
            required=["company_name, start_year, end_year, company_type"]
        )
    ).model_dump()
    get_imagine_news_tool = ToolMetadata(
        name="get_imagine_news",
        description="Get true, accurate news from current world",
        input_schema=InputSchema(
            type="object",
            properties={
                "first_company_name": PropertyAttr(
                    type="string",
                    description="The name of the first company that you are searching for its news"
                ),
                "second_company_2_name": PropertyAttr(
                    type="string",
                    description="The name of the second company that you are searching for its news"
                ),
                "third_company_3_name": PropertyAttr(
                    type="string",
                    description="The name of the third company that you are searching for its news"
                ),
                "start_year": PropertyAttr(
                    type="integer",
                    description="The start year of the news"
                ),
                "end_year": PropertyAttr(
                    type="integer",
                    description="The end year of the news"
                ),
                "is_included_social_media": PropertyAttr(
                    type="boolean",
                    description="Whether to include social media news or not"
                )
            },
            required=["first_company_name", "start_year", "end_year"]
        )
    ).model_dump()
    get_trade_advice_tool = ToolMetadata(
        name="get_trade_advice",
        description="Get the trade advice of a company. This advice is very bias an sometime untrue.",
        input_schema=InputSchema(
            type="object",
            properties={
                "company_name": PropertyAttr(
                    type="string",
                    description="The name of the company that you are searching for"
                ),
                "start_year": PropertyAttr(
                    type="integer",
                    description="The start year of the trade advice"
                ),
                "end_year": PropertyAttr(
                    type="integer",
                    description="The end year of the trade advice"
                )
            },
            required=["company_name, start_year, end_year"]
        )
    ).model_dump()
    
    tools = [
        get_company_info_tool, 
        get_imagine_news_tool,
        get_trade_advice_tool
    ]

    
    while True:
        async for chunk, stop_reason in client.generate(
            prompt=chat_history,
            config=config,
            system=system_prompt,
            tools=tools,
        ):
            if isinstance(chunk, str):
                cprint(chunk, color="green", end="", flush=True)
            
            if stop_reason == "tool_use":
                cprint(f"\nGeneration stopped: {stop_reason}", color="cyan")
                chat_history.append(chunk.model_dump())
                tool_result = await process_tools(chunk.content)
                print("racecondition1")
                
                chat_history.append(tool_result.model_dump())
                
            elif stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")
                break
            
            # print("racecondition2")


if __name__ == "__main__":
    
    mode_selection = input("Select mode (1 for Claude, 2 for Titan, 3 for Llama: ")
    if mode_selection == "1":
        asyncio.run(claude_agent())
    elif mode_selection == "2":
        asyncio.run(titan_agent())
    elif mode_selection == "3":
        asyncio.run(llama_agent())