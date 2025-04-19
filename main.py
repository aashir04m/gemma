from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiohttp
import os
import httpx
import json
import time
import logging
import sqlite3
from datetime import datetime
from typing import List, Optional
import contextlib
import pathlib
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RunPod Ollama API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (make sure this directory exists)
static_dir = os.path.join(os.getcwd(), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Database setup - use absolute path in a directory we know exists and is writable
db_dir = os.path.join(os.getcwd(), "data")
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

DATABASE_URL = os.path.join(db_dir, "chat_history.db")
logger.info(f"Database path: {DATABASE_URL}")

# Create database tables - use direct execution of SQL to ensure it works
def init_db():
    try:
        logger.info(f"Initializing database at {DATABASE_URL}")
        
        # Remove database if it exists but is empty or corrupted
        if os.path.exists(DATABASE_URL) and os.path.getsize(DATABASE_URL) == 0:
            logger.warning("Removing empty database file")
            os.remove(DATABASE_URL)
        
        # Execute SQL directly using sqlite3 command-line tool to guarantee it works
        sql_commands = '''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT 'New Conversation',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
        
        -- Insert a test conversation if none exist
        INSERT INTO conversations (title) 
        SELECT 'Welcome Conversation' 
        WHERE NOT EXISTS (SELECT 1 FROM conversations LIMIT 1);
        '''
        
        # Try using Python API first
        try:
            conn = sqlite3.connect(DATABASE_URL)
            conn.executescript(sql_commands)
            conn.commit()
            
            # Verify tables were created
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            logger.info(f"Created database tables: {table_names}")
            
            if 'conversations' not in table_names or 'messages' not in table_names:
                raise Exception("Tables weren't created properly")
                
            conn.close()
        except Exception as e:
            logger.warning(f"Error during Python-based DB initialization: {str(e)}")
            logger.info("Falling back to sqlite3 command line")
            
            # If Python API fails, try using the sqlite3 command-line tool
            with open("/tmp/init_db.sql", "w") as f:
                f.write(sql_commands)
            
            try:
                result = subprocess.run(
                    ["sqlite3", DATABASE_URL, ".read /tmp/init_db.sql"],
                    capture_output=True, text=True, shell=True
                )
                if result.returncode != 0:
                    logger.error(f"sqlite3 command error: {result.stderr}")
                    raise Exception(f"sqlite3 command failed: {result.stderr}")
                logger.info("Database initialized via sqlite3 command line")
            except Exception as cmd_error:
                logger.error(f"Error running sqlite3 command: {str(cmd_error)}")
                
                # Last resort: try direct file write with the exact SQL that SQLite needs
                try:
                    # Create an empty database file
                    with open(DATABASE_URL, 'wb') as f:
                        # SQLite database header (16 bytes)
                        f.write(b'SQLite format 3\0')
                        # Padding to create a valid empty database
                        f.write(b'\0' * 4080)
                    logger.info("Created empty database file as last resort")
                    
                    # Now try connecting and creating tables again
                    conn = sqlite3.connect(DATABASE_URL)
                    conn.executescript(sql_commands)
                    conn.commit()
                    conn.close()
                    logger.info("Successfully created tables after creating empty DB file")
                except Exception as last_error:
                    logger.error(f"Final attempt failed: {str(last_error)}")
                    return False
        
        # Double-check if the database has tables now
        try:
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            logger.info(f"Final database tables check: {table_names}")
            
            if 'conversations' not in table_names or 'messages' not in table_names:
                logger.error("Tables still missing after initialization attempts")
                conn.close()
                return False
                
            conn.close()
            return True
        except Exception as check_error:
            logger.error(f"Error during final table check: {str(check_error)}")
            return False
            
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        return False

# Database connection function
def get_db():
    try:
        conn = sqlite3.connect(DATABASE_URL, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Request body models
class ChatRequest(BaseModel):
    model: str = "gemma3:27b"  # Default to Gemma 3 27B
    messages: list
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    conversation_id: Optional[int] = None  # Added for tracking conversation

class PullModelRequest(BaseModel):
    name: str
    insecure: bool = False

class ConversationCreate(BaseModel):
    title: str = "New Conversation"

class ConversationUpdate(BaseModel):
    title: str

class Message(BaseModel):
    role: str
    content: str

# RunPod health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/")
async def serve_ui():
    ui_path = os.path.join(static_dir, "gemma_UI_with_formated.html")
    
    # Check if the file exists
    if not os.path.exists(ui_path):
        # Create default UI file
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Gemma Chat</title>
</head>
<body>
    <h1>Gemma Chat</h1>
    <p>UI file not found. Please place your UI file at static/gemma_UI_with_formated.html</p>
</body>
</html>"""
        with open(ui_path, "w") as f:
            f.write(html_content)
    
    return FileResponse(ui_path)

# RunPod serverless compatible handler
@app.post("/")
async def runpod_handler(request: Request):
    body = await request.json()
    input_data = body.get("input", {})
    
    # Check if this is a chat request
    if "messages" in input_data and "model" in input_data:
        chat_request = ChatRequest(**input_data)
        return await handle_chat(chat_request)
    
    # Default response if request type not recognized
    return {"error": "Unrecognized request type"}

# Ollama chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    response = await handle_chat(request)
    
    # Save conversation and message if conversation_id is provided
    if request.conversation_id:
        # Save user message
        user_message = request.messages[-1]  # Get the last user message
        
        # Add to background tasks to avoid blocking response
        background_tasks.add_task(
            save_chat_message, 
            request.conversation_id, 
            user_message["role"], 
            user_message["content"]
        )
        
        # For non-streaming responses, save assistant's response
        if not request.stream:
            # Extract assistant's response from the completed response
            assistant_message = response.get("message", {}).get("content", "")
            background_tasks.add_task(
                save_chat_message,
                request.conversation_id,
                "assistant",
                assistant_message
            )
    
    return response

async def handle_chat(request: ChatRequest):
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
    ollama_url = f"http://{ollama_host}:11434/api/chat"
    req_json = request.dict(exclude={"conversation_id"})  # Remove conversation_id before sending to Ollama
    
    if request.stream:
        async def stream_ollama():
            async with aiohttp.ClientSession() as session:
                async with session.post(ollama_url, json=req_json) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        yield json.dumps({"error": error_text}).encode()
                    else:
                        # For streaming, we'll collect the full response to save later
                        full_content = ""
                        conversation_id = request.conversation_id
                        
                        async for line in resp.content:
                            yield line
                            
                            # If we have a conversation_id, collect the content for saving
                            if conversation_id:
                                try:
                                    data = json.loads(line)
                                    if data.get("message", {}).get("content"):
                                        full_content += data["message"]["content"]
                                    
                                    # When the stream is done, save the complete assistant message
                                    if data.get("done", False):
                                        # This will be executed in a background task
                                        BackgroundTasks.add_task(
                                            save_chat_message, 
                                            conversation_id, 
                                            "assistant", 
                                            full_content
                                        )
                                except Exception as e:
                                    logger.error(f"Error processing stream: {str(e)}")
                                    pass

        return StreamingResponse(stream_ollama(), media_type="application/x-ndjson")
    else:
        async with httpx.AsyncClient() as client:
            response = await client.post(ollama_url, json=req_json, timeout=None)
            return JSONResponse(content=response.json())

async def save_chat_message(conversation_id: int, role: str, content: str):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content)
        )
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving message: {str(e)}")

# Endpoint to pull models
@app.post("/api/pull")
async def pull_model(request: PullModelRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(download_model, request.name, request.insecure)
    return {"status": "Model download started", "model": request.name}

# Function to download model in background
async def download_model(model_name: str, insecure: bool = False):
    logger.info(f"Starting download of model: {model_name}")
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
    ollama_url = f"http://{ollama_host}:11434/api/pull"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                ollama_url, 
                json={"name": model_name, "insecure": insecure},
                timeout=None
            )
            logger.info(f"Model download response: {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")

# List available models
@app.get("/api/models")
async def list_models():
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
    ollama_url = f"http://{ollama_host}:11434/api/tags"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(ollama_url)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Chat history endpoints
@app.post("/api/conversations", status_code=201)
async def create_conversation(conversation: ConversationCreate):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (title) VALUES (?)",
            (conversation.title,)
        )
        conn.commit()
        
        # Get the created conversation
        conversation_id = cursor.lastrowid
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        result = cursor.fetchone()
        
        response_data = {
            "id": result["id"],
            "title": result["title"],
            "created_at": result["created_at"],
            "updated_at": result["updated_at"]
        }
        
        conn.close()
        return response_data
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/conversations")
async def list_conversations():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        )
        results = cursor.fetchall()
        
        response_data = [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            for row in results
        ]
        
        conn.close()
        return response_data
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get conversation
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        conversation = cursor.fetchone()
        
        if not conversation:
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        cursor.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        messages = cursor.fetchall()
        
        response_data = {
            "id": conversation["id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": [
                {
                    "id": msg["id"],
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"]
                }
                for msg in messages
            ]
        }
        
        conn.close()
        return response_data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: int, conversation: ConversationUpdate):
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if conversation exists
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update conversation
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation.title, conversation_id)
        )
        conn.commit()
        
        # Get updated conversation
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        result = cursor.fetchone()
        
        response_data = {
            "id": result["id"],
            "title": result["title"],
            "created_at": result["created_at"],
            "updated_at": result["updated_at"]
        }
        
        conn.close()
        return response_data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error updating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if conversation exists
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete conversation (will cascade delete messages)
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
        
        return {"message": f"Conversation {conversation_id} deleted"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Debug endpoint to check database status
@app.get("/api/debug/database")
async def check_database():
    """Debug endpoint to check if database is properly initialized"""
    try:
        # First, check if the database file exists
        db_exists = os.path.exists(DATABASE_URL)
        db_size = os.path.getsize(DATABASE_URL) if db_exists else 0
        
        # Try to initialize the database if it doesn't exist or is empty
        if not db_exists or db_size == 0:
            logger.info("Database doesn't exist or is empty, attempting initialization")
            init_db()
        
        # Now check the database status
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        table_names = [table['name'] for table in tables]
        
        # If tables don't exist, try to create them one more time
        if 'conversations' not in table_names or 'messages' not in table_names:
            logger.warning("Tables missing, attempting to create them")
            conn.close()
            init_db()
            
            # Check again
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table['name'] for table in tables]
        
        # Count records in main tables
        conversation_count = 0
        message_count = 0
        
        if 'conversations' in table_names:
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            conversation_count = cursor.fetchone()['count']
        
        if 'messages' in table_names:
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            message_count = cursor.fetchone()['count']
        
        # Get SQLite version for diagnostic purposes
        cursor.execute("SELECT sqlite_version()")
        sqlite_version = cursor.fetchone()[0]
        
        # Get database file permissions
        try:
            import stat
            file_stat = os.stat(DATABASE_URL)
            file_permissions = stat.filemode(file_stat.st_mode)
        except Exception as perm_error:
            file_permissions = f"Error getting permissions: {str(perm_error)}"
        
        # Check write access to the database directory
        dir_writable = os.access(db_dir, os.W_OK)
        file_writable = os.access(DATABASE_URL, os.W_OK) if db_exists else False
        
        response_data = {
            "database_path": DATABASE_URL,
            "database_exists": db_exists,
            "db_size_bytes": db_size,
            "tables": table_names,
            "tables_status": {
                "conversations_exists": 'conversations' in table_names,
                "messages_exists": 'messages' in table_names,
            },
            "records": {
                "conversation_count": conversation_count,
                "message_count": message_count,
            },
            "system_info": {
                "sqlite_version": sqlite_version,
                "file_permissions": file_permissions,
                "dir_writable": dir_writable,
                "file_writable": file_writable,
                "current_working_dir": os.getcwd(),
            }
        }
        
        conn.close()
        return response_data
    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
        return {
            "error": str(e),
            "database_path": DATABASE_URL,
            "database_exists": os.path.exists(DATABASE_URL),
            "db_size_bytes": os.path.getsize(DATABASE_URL) if os.path.exists(DATABASE_URL) else 0,
        }

# Fix database endpoint
@app.post("/api/debug/fix-database")
async def fix_database():
    """Endpoint to force database initialization"""
    try:
        # If database exists but is empty or corrupted, remove it
        if os.path.exists(DATABASE_URL):
            os.remove(DATABASE_URL)
            logger.info(f"Removed existing database file: {DATABASE_URL}")
        
        # Initialize database
        success = init_db()
        
        if success:
            # Verify tables
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table['name'] for table in tables]
            conn.close()
            
            return {
                "success": True,
                "message": "Database successfully initialized",
                "tables": table_names
            }
        else:
            return {
                "success": False,
                "message": "Database initialization failed, check server logs"
            }
    except Exception as e:
        logger.error(f"Error fixing database: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

# Startup event
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting FastAPI application")
    
    # Initialize database
    init_db()
    
    # Wait for Ollama to be ready
    ollama_ready = False
    max_retries = 10
    retry_count = 0
    
    while not ollama_ready and retry_count < max_retries:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    ollama_ready = True
                    logger.info("Ollama is ready")
                else:
                    logger.info(f"Ollama not ready yet, status: {response.status_code}")
        except Exception as e:
            logger.info(f"Waiting for Ollama to start: {str(e)}")
        
        if not ollama_ready:
            retry_count += 1
            time.sleep(5)
    
    if not ollama_ready:
        logger.warning("Ollama service not ready after maximum retries")
    
    yield  # This is where the app runs
    
    # Shutdown logic (if any) would go here