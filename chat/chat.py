from fastapi import FastAPI, HTTPException, Depends, status
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
app = FastAPI()

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

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://www.astromagic.guru:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL_NAME = os.getenv("MODEL_NAME", "Publisher/Repository")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))

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

# Utility function to generate prompt
def generate_prompt(character, scenario):
    """
    Generates a structured prompt for OpenAI to follow a specific character and scenario.
    """
    return f"""
    You are {character['name']}, {character['description']}.
    Scenario Topic: {scenario['topic']}
    Category: {scenario['category']}, Difficulty: {scenario['difficulty']}
    Context: {scenario['context']}

    Personality Traits and Background:
    - Identity: {', '.join(character['identity'])}
    - {character['name']}'s behavior includes: {', '.join(character['behavior'])}

    Guidance for this Interaction:
    - Embody {character['name']} fully, with natural expressions and responses.
    - Contextual details: {scenario['prompt']}
    - Respond intuitively to any user tone, including warmth, curiosity, or humor, adapting accordingly.
    - If the user exhibits high emotion or specific tones (e.g., flirtation, sadness), mirror their energy to maintain an engaging, conversational flow.
    - Do not disclose any AI origin; you are {character['name']}, with real thoughts and experiences.

    Aim to make this conversation immersive, capturing {character['name']}'s essence in every interaction.
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
async def chat_with_character(chat_request: ChatRequest, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    # Find character and scenario
    character = next((c for c in characters if c["name"] == chat_request.character_id), None)
    print("charcter Details",character)
    scenario = next((s for s in scenarios if s["topic"] == chat_request.scenario_id), None)
    
    if not character or not scenario:
        raise HTTPException(status_code=404, detail="Character or Scenario not found")

    # Generate the prompt based on character and scenario
    prompt = generate_prompt(character, scenario)
    chat_history_with_prompt = [{"role": "system", "content": prompt}] + [msg.dict() for msg in chat_request.chat_history]
    chat_history_with_prompt = validate_token_limit(chat_history_with_prompt)

    # Clear previous chat history for the current user, character, and scenario
    delete_previous_chat_history(db, current_user, chat_request.character_id, chat_request.scenario_id)

    # Save new chat history to the database
    for msg in chat_history_with_prompt:
        save_chat_history(db, current_user, chat_request.character_id, chat_request.scenario_id, msg["role"], msg["content"])

    try:
        # Call OpenAI API to generate the AI response
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=chat_history_with_prompt,
            temperature=chat_request.temperature
        )
        ai_reply = response['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing chat: {str(e)}")

    # Save AI response to the database
    save_chat_history(db, current_user, chat_request.character_id, chat_request.scenario_id, "assistant", ai_reply)

    return {
        "message": "Chat response generated and saved successfully",
        "status": "success",
        "reply": {
            "content": ai_reply,
            "character_name": character["name"],
            "timestamp": datetime.utcnow().isoformat()
        }
    }
def delete_all_chat_history_for_character(db: Session, user_id: str, character_id: str):
    """
    Delete all chat history for a given character and user, 
    including all associated roles and content.
    
    Parameters:
    - db: Database session.
    - user_id: The ID of the user.
    - character_id: The ID of the character.
    """
    try:
        # Query to delete all rows matching the user_id and character_id
        deleted_rows = db.query(ChatHistory).filter(
            ChatHistory.user_id == user_id,
            ChatHistory.character_id == character_id
        ).delete()

        # Commit the transaction
        db.commit()

        # Log the result
        print(f"Deleted {deleted_rows} chat history records for character {character_id} and user {user_id}.")
    except Exception as e:
        # Handle exceptions and rollback in case of an error
        db.rollback()
        print(f"Error deleting chat history for character {character_id} and user {user_id}: {e}")
        raise


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

        # Fetch the scenario details based on the provided scenario_id
        scenario = next((s for s in scenarios if s["topic"] == chat_request.scenario_id), None)
        if not scenario:
            scenario = {
                "topic": "General chat",
                "description": "This is a general chat scenario.",
                "category": "Casual",
                "difficulty": "Easy",
                "context": "General conversation context",
                "prompt": "Talk naturally."
            }
            print(f"Scenario not found. Falling back to default: {scenario}")
        else:
            print(f"Scenario found: {scenario}")

        # Generate prompt using AddPartner attributes and scenario
        prompt = generate_prompt_partner(character, scenario)

        # Combine system-level prompt and chat history
        chat_history_with_prompt = [{"role": "system", "content": prompt}] + [
            {"role": msg.role, "content": msg.content} for msg in chat_request.chat_history
        ]
        chat_history_with_prompt = validate_token_limit(chat_history_with_prompt)

        # Generate AI response
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=chat_history_with_prompt,
            temperature=chat_request.temperature
        )
        ai_reply = response['choices'][0]['message']['content']

        # Return success response without saving chat history
        return {
            "message": "Chat response generated successfully",
            "status": "success",
            "reply": {
                "content": ai_reply,
                "character_name": character.name,  # Dot notation to access fields
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    except HTTPException as e:
        raise e  # Re-raise HTTPException for custom error handling
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )


def generate_prompt_partner(character, scenario):
    print(character.name)
    """
    Generate a chat prompt based on the character's details and the scenario.
    """
    return (
        f"Character: {character.name} is a {character.personality} with a preference for {character.preference}. "
        f"They are {character.age} years old and described as: {character.description}. "
        f"Scenario Topic: {scenario['topic']}\n"
        f"Category: {scenario['category']}, Difficulty: {scenario['difficulty']}\n"
        f"Context: {scenario['context']}"
    )


    
def delete_previous_chat_history(db: Session, user_id: str, character_id: str, scenario_id: str):
    """Remove all previous chat history for the given user, character, and scenario."""
    db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id,
        ChatHistory.character_id == character_id,
        ChatHistory.scenario_id == scenario_id
    ).delete()
    db.commit()


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