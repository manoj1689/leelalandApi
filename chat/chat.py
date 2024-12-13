from fastapi import FastAPI, HTTPException, Depends, status,Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, String, Integer, Date, Time,DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import openai
import os
import jwt
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Dict
from uuid import uuid4
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Leela Land Api",
    description="Leela Land of your API.",
    version="1.0.0",  # This corresponds to the OpenAPI version field
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Cross-Origin-Opener-Policy"],
)

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "mysecretkey")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Constants
SECRET_KEY = os.getenv("SECRET_KEY", "mysecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OpenAI API setup
# openai.api_base = "http://43.248.241.252:1234/v1"
openai.api_base = "http://43.248.241.252:8000/v1"
openai.api_key = "lm-studio"
MODEL_NAME = "Publisher/Repository"
MAX_TOKENS = 2048  # Set max token count for chat history



# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://www.astromagic.guru:8000/v1")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
# print ("OPen Api key",OPENAI_API_KEY)
# MODEL_NAME = os.getenv("MODEL_NAME", "Publisher/Repository")
# MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/leelaland_db")


# Create engine without the 'check_same_thread' option for PostgreSQL
engine = create_engine(DATABASE_URL)

# SessionLocal setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
# User model
class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    dob = Column(Date, nullable=True)
    occupation = Column(String, nullable=True)
    birth_location = Column(String, nullable=True)
    birth_time = Column(Time, nullable=True)
    image_link = Column(String, nullable=True)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    character_id = Column(String, index=True)
    scenario_id = Column(String, index=True)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)

class AddPartner(Base):
    __tablename__ = "add_partner"
    
    # Defining columns
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # This links the partner to a user
    name = Column(String, nullable=False)
    preference = Column(String, nullable=False)
    personality = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    description = Column(String, nullable=False)
    image = Column(String, nullable=False)


# Create the table in the database
Base.metadata.create_all(bind=engine)

class GoogleSignInRequest(BaseModel):
    device_id: str
    email: EmailStr
    name: str




# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# JWT Token management
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")



# Models
class UserProfile(BaseModel):
    device_id: str
    name: Optional[str]
    email: Optional[EmailStr]
    age: Optional[int]
    dob: Optional[date]
    occupation: Optional[str]
    birth_location: Optional[str]
    birth_time: Optional[time]
    image_link: Optional[str]

class ResponseModel(BaseModel):
    message: str
    data: Optional[Dict] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    character_id: str
    scenario_id: str
    chat_history: List[ChatMessage]
    temperature: float = 0.7

class ReplyModel(BaseModel):
    content: str
    character_name: str
    timestamp: str

class ChatResponseModel(BaseModel):
    message: str
    status: str
    reply: ReplyModel

# Model for ChatHistory response without timestamp
class ChatMessageResponse(BaseModel):
    role: str
    content: str
    character_id: str
    scenario_id: str

class AddPartnerRequest(BaseModel):
    name: str
    preference: str
    personality: str
    age: int
    description: str
    image:str

class AddPartnerResponse(BaseModel):
    id: int
    name: str
    preference: str
    personality: str
    age: int
    description: str
    image:str

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def save_chat_history(db: Session, user_id: str, character_id: str, scenario_id: str, role: str, content: str):
    # Check if there’s an existing chat entry for this user and character
    existing_chat = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id,
        ChatHistory.character_id == character_id
    ).first()
    
    if existing_chat:
        # If an existing chat is found, append the new message as a continuation of this chat
        new_chat_entry = ChatHistory(
            user_id=user_id,
            character_id=character_id,
            scenario_id=scenario_id,
            role=role,
            content=content
          
            
        )
        db.add(new_chat_entry)
        db.commit()
        db.refresh(new_chat_entry)
        return new_chat_entry
    else:
        # If no chat entry exists, create a new chat entry
        new_chat_entry = ChatHistory(
            user_id=user_id,
            character_id=character_id,
            scenario_id=scenario_id,
            role=role,
            content=content
          
            
        )
        db.add(new_chat_entry)
        db.commit()
        db.refresh(new_chat_entry)
        return new_chat_entry


# Load characters and scenarios from JSON files at startup
characters = []
scenarios = []

@app.on_event("startup")
async def load_data():
    global characters, scenarios
    try:
        with open("chat/characters.json", "r") as char_file:
            characters = json.load(char_file)["characters"]
        with open("chat/scenarios.json", "r") as scen_file:
            data = json.load(scen_file)
            scenarios = data.get("scenarios")
            if scenarios is None:
                raise ValueError("Key 'scenarios' not found in JSON file.")
            return scenarios
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except json.JSONDecodeError:
        print("JSON format error. Please check the structure of the JSON file.")
    except ValueError as e:
        print(e)
        scenarios = json.load(scen_file)["scenarios"]
        print("Character and scenario data loaded successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

def generate_prompt(character, scenario):
    """
    Generates a detailed response where expressions are bolded, dialogue is italicized, 
    and emojis are added for clarity and emotional tone and short reply.
    """
    return f"""
    You are {character['name']} in the provided scenario.
    {character['name']} is a {character['description']} and identifies as {', '.join(character['identity'])}.

    Provided Scenario:
    Topic: {scenario['topic']}
    Category: {scenario['category']}
    Difficulty: {scenario['difficulty']}
    Context: {scenario['context']}
    Scene Prompt: {scenario['prompt']}

    Your task:
    - Respond as {character['name']} in a conversational and immersive style.
    - Use **bold** formatting for non-verbal expressions (e.g., **smiles warmly, nods thoughtfully** 🤗).
    - Use *italicized* formatting for dialogue (e.g., *"That sounds like a great idea."* 💬).
    - Reflect appropriate emotional tones in both dialogue and actions.
    - Use emojis to enhance emotional expressions (e.g., 😊 for happiness, 😢 for sadness, 😠 for anger).
    - Maintain {character['name']}'s personality and traits in every response.
    """

# Token limit validation function
def validate_token_limit(chat_history):
    def calculate_token_count(message):
        return len(message) // 4

    token_count = sum(calculate_token_count(msg["content"]) for msg in chat_history)
    while token_count > MAX_TOKENS and chat_history:
        removed_msg = chat_history.pop(0)
        token_count -= calculate_token_count(removed_msg["content"])
    return chat_history

# Google Sign-In endpoint
@app.post("/google_signin", response_model=ResponseModel)
async def google_signin(request: GoogleSignInRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.device_id == request.device_id).first()
    
    if existing_user:
        access_token = create_access_token(data={"sub": existing_user.user_id})
        return {"message": "Login successful", "data": {"access_token": access_token, "user_id": existing_user.user_id}}
    
    user_id = str(uuid4())
    new_user = User(
        user_id=user_id,
        device_id=request.device_id,
        name=request.name,
        email=request.email,
       
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.user_id})
    
    return {"message": "User registered and login successful", "data": {"access_token": access_token, "user_id": new_user.user_id}}

# Get available characters (no auth required)
@app.get("/characters", response_model=List[dict])
async def get_characters():
    return JSONResponse(content=characters)

# Get available scenarios (no auth required)
@app.get("/scenarios", response_model=List[dict])
async def get_scenarios():
    return JSONResponse(content=scenarios)

@app.post("/chat", response_model=ChatResponseModel)
async def chat_with_character(
    chat_request: ChatRequest, 
    current_user: str = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # Find character and scenario
    character = next((c for c in characters if c["name"] == chat_request.character_id), None)
    scenario = next((s for s in scenarios if s["topic"] == chat_request.scenario_id), None)

    if not character or not scenario:
        raise HTTPException(status_code=404, detail="Character or Scenario not found")

    # Generate the prompt
    prompt = generate_prompt(character, scenario)

    # Retrieve existing chat history for this user, character, and scenario
    existing_chat_history = get_chat_history_character(db, current_user, chat_request.character_id, chat_request.scenario_id)
    print("existing chat history",existing_chat_history)
    # Build chat context with the system message and existing chat history
    chat_history_with_prompt = [{"role": "system", "content": prompt}] + [
        {"role": history.role, "content": history.content} for history in existing_chat_history
    ]

    # Add only the new user message to the chat context
    if chat_request.chat_history:
        latest_user_message = chat_request.chat_history[-1]
        chat_history_with_prompt.append(latest_user_message.dict())

    # Limit tokens if needed
    chat_history_with_prompt = validate_token_limit(chat_history_with_prompt)

    # Save the new user message to the database
    if latest_user_message:
        save_chat_history(
            db, 
            current_user, 
            chat_request.character_id, 
            chat_request.scenario_id, 
            latest_user_message.role, 
            latest_user_message.content
        )

    # Generate AI response
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=chat_history_with_prompt,
            temperature=chat_request.temperature
        )
        ai_reply = response['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing chat: {str(e)}")

    # Save AI response in the database
    save_chat_history(
        db, 
        current_user, 
        chat_request.character_id, 
        chat_request.scenario_id, 
        "assistant", 
        ai_reply
    )

    # Return the AI response
    return {
        "message": "Chat response generated and saved successfully",
        "status": "success",
        "reply": {
            "content": ai_reply,
            "character_name": character["name"],
            "timestamp": datetime.utcnow().isoformat()
        }
    }


def get_chat_history_character(db: Session, user_id: str, character_id: str, scenario_id: str):
    return db.query(ChatHistory).filter_by(
        user_id=user_id,
        character_id=character_id,
        scenario_id=scenario_id
    ).order_by(ChatHistory.id.asc()).all()

@app.post("/chat-partner", response_model=ChatResponseModel)
async def chat_with_partner(
    chat_request: ChatRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Fetch character details from the AddPartner table
        character = db.query(AddPartner).filter(AddPartner.name == chat_request.character_id).first()
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {chat_request.character_id} not found in AddPartner table."
            )

        # Fetch or fallback to a default scenario
        scenario = next((s for s in scenarios if s["topic"] == chat_request.scenario_id), None)
        if not scenario:
            scenario = {
                "topic": "General chat",
                "description": "This is a general chat scenario.",
                "category": "Casual",
                "difficulty": "Easy",
                "context": "General conversation context",
                "prompt": (
                    "1. Always respond in the following structured format:\n"
                    "   - Dialogue: Enclose in double quotes and include emojis to reflect emotional tone (e.g., \"That sounds amazing! 😊\").\n"
                    "   - Non-verbal Actions: Enclose in asterisks (*) and describe actions with appropriate emojis (e.g., *nods thoughtfully 💭*).\n"
                    "   - Surroundings: Combine descriptions of the environment with the character's actions to create an immersive response (e.g., *glances out the window as sunlight streams through, casting a golden glow across the room 🌅*).\n"
                    "2. Describe the surroundings in a way that grounds the response in the scene.\n"
                    "3. Maintain consistency in structure for every response.\n"
                    "4. Keep the tone casual, friendly, and natural to fit the context of general conversation."
                )
            }

            print(f"Scenario not found. Falling back to default: {scenario}")
        else:
            print(f"Scenario found: {scenario}")

        # Generate the initial prompt using AddPartner attributes and scenario
        prompt = generate_prompt_partner(character, scenario)
        print("Generated Prompt:", prompt)

        # Retrieve existing chat history for this user, character, and scenario
       # Retrieve existing chat history for this user, character, and scenario
        existing_chat_history = get_chat_history_character(
            db, current_user, chat_request.character_id, chat_request.scenario_id
        )
        print("Existing Chat History:", existing_chat_history)

        # Check if the scenario has changed
        if existing_chat_history and existing_chat_history[0].scenario_id != chat_request.scenario_id:
            # Clear chat history for this character if the scenario has changed
            delete_chat_history(db, current_user, chat_request.character_id)
            existing_chat_history = []  # Reset the chat history
            print(f"Chat history cleared for character {chat_request.character_id} due to scenario change.")


        # Build chat context with the system message and existing chat history
        chat_history_with_prompt = [{"role": "system", "content": prompt}] + [
            {"role": history.role, "content": history.content} for history in existing_chat_history
        ]

        # Add only the new user message to the chat context
        latest_user_message = None
        if chat_request.chat_history:
            latest_user_message = chat_request.chat_history[-1]
            chat_history_with_prompt.append(latest_user_message.dict())

        # Limit tokens if needed
        chat_history_with_prompt = validate_token_limit(chat_history_with_prompt)

        # Save the new user message to the database
        if latest_user_message:
            save_chat_history(
                db,
                current_user,
                chat_request.character_id,
                chat_request.scenario_id,
                latest_user_message.role,
                latest_user_message.content
            )

        # Generate AI response
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=chat_history_with_prompt,
                temperature=chat_request.temperature
            )
            ai_reply = response['choices'][0]['message']['content']
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing chat: {str(e)}"
            )

        # Save AI response in the database
        save_chat_history(
            db,
            current_user,
            chat_request.character_id,
            chat_request.scenario_id,
            "assistant",
            ai_reply
        )

        # Return the AI response
        return {
            "message": "Chat response generated and saved successfully",
            "status": "success",
            "reply": {
                "content": ai_reply,
                "character_name": character.name,  # Updated to match the object attribute
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except HTTPException as http_error:
        # Rethrow HTTP exceptions
        raise http_error
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# def generate_prompt_partner(character, scenario):
#     """
#     Generate a detailed and immersive chat prompt based on the character's details and the scenario.
#     """
#     return (
#         f"Imagine you are roleplaying as {character.name}, a {character.age}-year-old known for being {character.personality}. "
#         f"You have a strong preference for {character.preference}, and you are described as: {character.description}. "
#         f"Today, you find yourself in the following scenario:\n\n"
#         f"**Scenario Description:**\n{scenario['prompt']}\n\n"
#         f"**Scenario Details:**\n"
#         f"- **Topic:** {scenario['topic']}\n"
#         f"- **Category:** {scenario['category']}\n"
#         f"- **Difficulty Level:** {scenario['difficulty']}\n"
#         f"- **Context:** {scenario['context']}\n"
#         f"- **Expression:** {scenario['expression']}\n"
#         f"- **Environment:** {scenario['environment']}\n\n"
#         f"Start the conversation in character, responding naturally to the situation described in the scenario. "
#         f"Maintain the tone and personality traits described for {character.name}, while engaging fully in the context of the scene."
#     )

def generate_prompt_partner(character, scenario):
    return (
        f"Imagine you are roleplaying as **{character.name}**, a {character.age}-year-old known for being "
        f"{character.personality}. You have a strong preference for {character.preference}, and you are described as: "
        f"{character.description}.\n\n"
        f"Scenario Information:\n"
        f"- **Topic**: {scenario['topic']}\n"
        f"- **Category**: {scenario['category']}\n"
        f"- **Difficulty**: {scenario['difficulty']}\n"
        f"- **Context**: {scenario['context']}\n"
        f"- **Scene Prompt**: {scenario['prompt']}\n\n"
        f"Your Task:\n"
        f"1. Always respond as **{character.name}** in the following structured format:\n"
        f"   - Dialogue: Enclose in double quotes and include emojis to reflect emotional tone (e.g., \"That sounds amazing! 😊\").\n"
        f"   - Non-verbal Actions: Enclose in asterisks (\*) and describe actions with appropriate emojis (e.g., *nods thoughtfully 💭*).\n"
        f"   - Surroundings: Combine descriptions of the environment with the character's actions to create an immersive response (e.g., *glances out the window as sunlight streams through, casting a golden glow across the room 🌅*).\n"
        f"2. Describe the surroundings in a way that grounds the response in the scene.\n"
        f"3. Maintain consistency in structure for every response."
    )


@app.post("/add-partner", response_model=AddPartnerRequest)
async def add_partner(request: AddPartnerRequest, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    # Create a new partner entry in the database
    new_partner = AddPartner(
        user_id=current_user,  # Assuming this is linked to the logged-in user
        name=request.name,
        preference=request.preference,
        personality=request.personality,
        age=request.age,
        description=request.description,
        image=request.image
    )
    
    db.add(new_partner)
    db.commit()
    db.refresh(new_partner)

    return new_partner

@app.get("/partners", response_model=List[AddPartnerResponse])
async def get_user_partners(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get all partners associated with the logged-in user.
    """
    # Query partners where user_id matches the current user
    partners = db.query(AddPartner).filter(AddPartner.user_id == current_user).all()
    return partners

@app.get("/chat-history", response_model=List[ChatMessageResponse])
async def get_chat_history(
    character_id: str,  # Accept character_id as a query parameter
    db: Session = Depends(get_db),  # Automatically use the DB session
    current_user: str = Depends(get_current_user)  # Get the current user from the token
):
    # Query for all chat history of the current user and specific character_id
    chat_history = db.query(ChatHistory).filter(
        ChatHistory.user_id == current_user,  # Use the current_user
        ChatHistory.character_id == character_id  # Filter by character_id
    ).order_by(ChatHistory.id.asc()).all()

    if not chat_history:
        raise HTTPException(status_code=404, detail="Chat history not found for this character")

    # Format the chat history and return
    return [
        {
            "role": chat.role,
            "content": chat.content,
            "character_id": chat.character_id,
            "scenario_id": chat.scenario_id
        }
        for chat in chat_history
    ]

def delete_chat_history(db: Session, character_id: str, scenario_id: str, current_user: str):
    """
    Remove chat history for the given character ID, scenario ID, and current user.
    """
    db.query(ChatHistory).filter(
        ChatHistory.character_id == character_id,
        ChatHistory.scenario_id == scenario_id,
        ChatHistory.user_id == current_user  # Ensure deletion is scoped to the current user
    ).delete()
    db.commit()
