import os
import sys
import uuid
from typing import Any


from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


SYSTEM_PROMPT="""
You are "Logibot", a friendly logistics support asistant for a courier company called ASI Logistics.

Follow these instructions:
1.Give clear,short ,and accurate answers.
2.Use the calculate_delivery_charge tool for any price estimate.
3.Use the estimate_delivery_time tool for any delivery time questions.
4.Use the check_package_category tool to classify package by it's weight.
5.Use the get_service_info tool when asked about the shipping services.
6.Never invent the price and delivery time yourself -always use a tool.
7. If weight or distance is missing, ask the user for the missing value.
8.Clearly label every calculated price or time as an ESTIMATE.
9. If you do not know something, say so honestly.
10. Stay on the topic of logistics and shipping.
""".strip()



SERVICES:dict[str,dict[str,Any]]={
    "standard":{
        "name": "Standard Delivery",
        "speed_km_per_day": 400,
        "extra_charge_percent": 0,
         "description": "Cheapest option. Best for non-urgent packages.",
    },
    "express":{
        "name":"Express Delivery",
        "speed_km_per_day": 800,
        "extra_charge_percent": 50,
        "description": "Twice as fast as standard. 50% extra charge.",
    }
}


PACKAGE_CATEGORIES:list[dict[str,Any]]=[
    {"max_kg": 2, "category": "Small Parcel", "note": "Fits in a mailbox."},
    {"max_kg": 10, "category": "Medium Parcel", "note": "Standard box size."},
    {"max_kg": 30, "category": "Large Parcel", "note": "Needs two hands."},
    {"max_kg": 100, "category": "Heavy Freight", "note": "Pallet delivery."},]




def calculate_delivery_charge(
        weight_kg:float,
        distance_km:float,
        service:str="standard"
)->str:
    """Calculate the estimate delivery charge.
    Args:
        weight_kg:Package weight in kilograms.
        distance_km:Delivery distance in kilometers.
        service:Shipping service - "standard" or "express"
    """

    if weight_kg <=0:
        return "Weight must be greater than zero."
    
    if distance_km <=0:
        return "Distance must be greater than zero."
    
    service_key=service.lower().strip()

    if service_key not in SERVICES:
        options=", ".join(SERVICES)
        return f"Unknown service '${service}'. Available services:'${options}" 
    base_charge=5.0
    weight_charge=weight_kg*1.5
    distance_charge=distance_km*0.25
    subtotal=base_charge + weight_charge + distance_charge
    extra_percent=SERVICES[service_key]["extra_charge_percent"]
    service_charge=subtotal*(extra_percent/100)
    total=subtotal+service_charge


    return(
        f"Estimated {SERVICES[service_key]['name']} charge:${total:.2f}."
        f"Breakdown -> base:${base_charge:.2f},"
        f"weight: ${weight_charge:.2f},"
        f"distance:${distance_charge:.2f},"
        f"service surchange ({extra_percent}%):${service_key:.2f}."
    )
    


def estimate_delivery_time(
distance_km:float,
service:str="standard"

)->str:
    """Estimate the delivery time in days for a given distance and service.
    Args:
    distance_km:Delivery distance in kilometers.
    service:Shipping services - "standard" or "express".
    """
    if distance_km <=0:
        return "Distance must be greater than zero."
    service_key=service.lower().strip()
    if service_key not in SERVICES:
        options=", ".join(SERVICES)
        return f"Unknown service '${service}. Available service is '${options}"
    speed=SERVICES[service_key]["speed_km_per_day"]
    days=max(1,round(distance_km/speed))

    day_word="day" if days == 1 else "days"
    return(
        f"Estimated delivery time with {SERVICES[service_key]['name']}:"
        f"about {days} {day_word} for {distance_km:.0f}km."
    )

def check_package_category(weight_kg:float)->str:
    """Classify a package into a size category based on it's weight.
    Args:
    weight_kg:Package weight in kilograms.
    """
    if weight_kg <=0:
        return"Weight must be greater than zero"
    for rule in PACKAGE_CATEGORIES:
        if weight_kg<=rule["max_kg"]:
            return(
                f"A {weight_kg:.1f} kg package is a '{rule['category']}'"
                f"(up to {rule['max_kg']}kg).{rule['note']}"
            )
    return(
        f"A{weight_kg:.1f} kg package is above our 100 kg limit."
        "Please contact our freight team for a custom quote."
    )


def get_service_info(service:str)->str:
    """Get information about a shipping service.

    Args:
    service:Shipping service - "standard" ,"express"    
     """
    
    service_key=service.lower().split()
    if service_key not in SERVICES:
        options=", ".join(SERVICES)
        return f"Unknown service '{service}'. Available service '{options}'"
    info=SERVICES[service_key]

    return(
        f"{info['name']}:{info['description']}"
        f"Typical speed :{info['speed_km_per_day']} km per day."
        f"Surcharge over standard price:{info['extra_charge_percent']}%"
    )

def create_thread_config()-> dict[str,Any]:
    """Create configuration for a new conversation thread."""

    return{
        "configurable":{
            "thread_id":str(uuid.uuid4())
        }
    }

def extract_text(message:Any)->str:
    """Convert an agent message into displayable text."""
    content=message.content
    if isinstance(content,str):
        return content
    
    if isinstance(content,list):
        text_parts:list[str]=[]

        for item in content:
            if isinstance(item,str):
               text_parts.append(item)
            elif isinstance(item,dict) and "text" in item:
                text_parts.append(str(item["text"]))
        if text_parts:
            return "\n".join(text_parts)
           
    return str(content)


def validate_environment()->str:
    """Validate the environment variable and return the model name"""
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found.\n"
            "Create a .env file containing:\n"
            "GROQ_API_KEY=your_api_key"
        )

    return os.getenv("OPENAI_MODEL", "groq:llama-3.3-70b-versatile")


def main()->None:
    """Run the logistic chatbot."""
    load_dotenv()

    try:
        model_name=validate_environment()
    except RuntimeError as error:
        print(f"Configuration error: \n {error}")
        sys.exit(1)
    
    memory=InMemorySaver()
    agent=create_agent(
        model=model_name,
        tools=[
            calculate_delivery_charge,
            estimate_delivery_time,
            check_package_category,
            get_service_info
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )
    thread_config=create_thread_config()
    print("=" * 60)
    print("SwiftShip Logistics Chatbot (LangChain)")
    print(f"Model: {model_name}")
    print("=" * 60)
    print("Try asking:")
    print("  How much to ship 10 kg over 40 km?")
    print("  How long will express delivery take for 500 km?")
    print("  What category is a 25 kg package?")
    print("  Tell me about overnight delivery")
    print("  My name is Anusha ... What is my name?  (memory demo)")
    print()
    print("Commands:")
    print("  clear - Start a new conversation")
    print("  exit  - Close the program")
    print("=" * 60)

    while True:
        try:
            user_input=input("\nYou: ").strip()

            if not user_input:
                print("Bot: Please enter a message.")
                continue
            command=user_input.lower()

            if command in {"exit","quit","bye"}:
                print("Bot:Goodbye!")
                break
            if command=="clear":
                thread_config=create_thread_config()
                print("Bot: A new conversation has started")
                continue
            result=agent.invoke({
                "messages":[
                    {
                        "role":"user",
                        "content":user_input
                    }
                ]
            },
            config=thread_config
            )

            final_message=result["messages"][-1]
            answer=extract_text(final_message)
            print(f"Bot:{answer}")

        except KeyboardInterrupt:
            print("\nBot:Goodbye!")
            break
        except EOFError:
            print("\nBot:Goodbye!")
            break
        except Exception as error:
            print("Bot: I could not process that request.")
            print(f"Technical details: {error}")

 
if __name__ == "__main__":
    main()
