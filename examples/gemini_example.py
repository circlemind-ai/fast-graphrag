import os

from fast_graphrag import GraphRAG, QueryParam
import asyncio
from fast_graphrag._utils import logger
from fast_graphrag._llm import GeminiLLMService, GeminiEmbeddingService #, VoyageAIEmbeddingService

WORKING_DIR="./book_example"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]

# Custom entity types for story analysis
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

# For VoyageAI Embeddings, higher rate limits
# from fast_graphrag._llm import VoyageAIEmbeddingService
# VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    
# For PDF Processing, uses langchain and PyMuPDF
#from langchain_community.document_loaders import PyMuPDFLoader
#async def process_pdf(file_path: str) -> str:
#    """Process PDFs with error handling"""
#    if not os.path.exists(file_path):
#        raise FileNotFoundError(f"PDF file not found: {file_path}")
#
#    try:
#        loader = PyMuPDFLoader(file_path)
#        pages = ""
#        for page in loader.lazy_load():
#            pages += page.page_content
#        return pages
#            
#    except Exception as e:
#        raise

# Text Processing Function
async def process_text(file_path: str) -> str:
    """Process text file with encoding handling"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    try:
        # Try UTF-8 first, fallback to other encodings
        encodings = ['utf-8', 'ascii', 'iso-8859-1', 'cp1252']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
                
        if text is None:
            raise UnicodeError("Failed to decode file with any supported encoding")
            
        # Clean and normalize text
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()
            
    except Exception as e:
        logger.exception("An error occurred:", exc_info=True)
        raise
    
async def streaming_query_loop(rag: GraphRAG):
    """Basic query loop for repeated questions"""
    print("\nStreaming Query Interface (type 'exit' to quit)")
    print("="*50)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() == 'exit':
                print("\nExiting chat...")
                break
                
            print("Assistant: ", end='', flush=True)
            
            try:
                # Higher token limits for Gemini context windows
                response = await rag.async_query(
                    query,
                    params=QueryParam(with_references=False, only_context=False, entities_max_tokens=250000, relations_max_tokens=200000, chunks_max_tokens=500000)
                )
                print(response.response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            logger.exception("An error occurred:", exc_info=True)
            continue
        
### fast-graphrag example for Gemini
async def main():     
    try:
        grag = GraphRAG(
            working_dir=WORKING_DIR,
            domain=DOMAIN,
            example_queries="\n".join(EXAMPLE_QUERIES),
            entity_types=ENTITY_TYPES,
            config=GraphRAG.Config(
                # Ensure necessary APIs have been enabled in Google Cloud, namely the Generative API
                # Supports Vertex usage via passing project_id and locatio, or using an 'Express' API key if available. More aggressive rate limiting will be required
                # Recommendation optionis using API keys from AI Studio, enabling Billing on the studio's project for much higher rate limits (2000 RPM for 2.0 Flash as of Feb 2025)
                llm_service = GeminiLLMService(
                    model="gemini-2.0-flash",
                    api_key=GEMINI_API_KEY,
                    temperature=0.7,
                    rate_limit_per_minute=True,
                    rate_limit_per_second=True,
                    max_requests_per_minute=1950,
                    max_requests_per_second=500
                ),
                embedding_service=GeminiEmbeddingService(
                    api_key=GEMINI_API_KEY,
                    max_requests_concurrent=100,
                ),
                # for Voyage AI embeddings with higher rate limits than Vertex
                #embedding_service=VoyageAIEmbeddingService(
                #    model="voyage-3-large",
                #    api_key=VOYAGE_API_KEY,
                #    embedding_dim=1024,  # the output embedding dim of the chosen model
                #),
            ),
        )

        file = await process_text("./book.txt")
        await grag.async_insert(file)
            
        await streaming_query_loop(grag)
    except Exception as e:
        logger.exception("An error occurred:", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
