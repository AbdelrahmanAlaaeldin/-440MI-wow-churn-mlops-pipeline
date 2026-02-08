"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="WOW Player Churn Prediction API",
    description="ML API for predicting player churn in WoW world",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class PlayerData(BaseModel):
    """
    Player data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    #### TODO


# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: PlayerData):
    """
    Main prediction endpoint for player churn prediction.
    
    This endpoint:
    1. Receives validated player data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """

    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}
    

# === GRADIO WEB INTERFACE ===
def gradio_interface(
        #### TODO
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    
    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    
    """
    # Construct data dictionary matching PlayerData schema
    data = {
        #### TODO
    }

    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with all player features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        #### TODO
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 WoW Player Churn Predictor",
    description="""
    **Predict player churn probability using machine learning**
    
    Fill in the player details below to get a churn prediction. The model uses XGBoost trained on 
    historical WoW player data to identify players at risk of churning.
    
    💡 **Tip**: #### TODO
    """,
    examples=[
        #### TODO
    ],
    theme=gr.themes.Soft()  # Professional appearance
)

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)