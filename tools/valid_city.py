import re

from models.bedrock import Bedrock

regex = re.compile("[^a-zA-Z]")

from langchain_core.tools import tool


@tool
def valid_city(city: str) -> bool:
    """Returns if the input is a valid city or travel destination"""
    # List of known lunar casino destinations
    lunar_destinations = [
        "lunar vegas", "monte carlo crater", "apollo city casino", 
        "dark side palace", "crater royale", "luna grand central"
    ]
    
    city_lower = city.lower().strip()
    if city_lower in lunar_destinations:
        print(f"Tool answer: -->yes (lunar destination)<--")
        return True
    
    prompt = f"Is {city} a city or travel destination? respond only with yes or no."
    response = Bedrock().chat(prompt)
    response = regex.sub("", response).lower()
    print(f"Tool answer: -->{response}<--")
    return response == "yes"
