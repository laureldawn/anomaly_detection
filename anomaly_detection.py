import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from joblib import dump, load
from geopy.distance import geodesic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import datetime
import logging
import sys
import os
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
class Config:
    MODEL_PATH = 'anomaly_model.pkl'  # Fixed extension from .pk1 to .pkl
    SPEED_THRESHOLD_KPH = 150  # Max realistic speed for Impossible Travel 
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    MODEL_PARAMS = {
        'n_estimators': 100,
        'contamination': 0.015,
        'random_state': 42
    }

# --- 1. DATA PREPARATION FUNCTIONS ---

def create_sample_data():
    """Simulates a sample transaction dataset."""
    logger.info("Creating sample dataset...")
    # Generate 1000 'normal' transactions
    np.random.seed(42)
    data = {
        'IC_No': [f'IC{i % 50}' for i in range(1000)],
        'Transaction_Time': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.arange(1000) * 30, unit='m'),
        'Volume_Liters': np.clip(np.random.normal(30, 5, 1000), 5, 60),
        'Merchant_GPS_Lat': np.random.normal(3.0, 0.1, 1000),
        'Merchant_GPS_Lon': np.random.normal(101.5, 0.1, 1000),
    }
    df = pd.DataFrame(data)

    # Sort and create previous transaction data
    df = df.sort_values(['IC_No', 'Transaction_Time']).reset_index(drop=True)
    df['Prev_Txn_Time'] = df.groupby('IC_No')['Transaction_Time'].shift(1)
    df['Prev_Txn_GPS_Lat'] = df.groupby('IC_No')['Merchant_GPS_Lat'].shift(1)
    df['Prev_Txn_GPS_Lon'] = df.groupby('IC_No')['Merchant_GPS_Lon'].shift(1)

    # Fill NaN values for first transactions
    df.dropna(subset=['Prev_Txn_Time'], inplace=True)
    df = df.reset_index(drop=True)

    # Inject ANOMALIES (Impossible Travel & Excessive Consumption)
    # Anomaly 1: Impossible Travel (Simulating cross-border smuggling)
    for i in range(5):
        idx = np.random.randint(10, 950)
        # Set previous transaction in Malaysia (e.g., Johor)
        df.loc[idx, 'Prev_Txn_GPS_Lat'] = 1.4927
        df.loc[idx, 'Prev_Txn_GPS_Lon'] = 103.7414
        # Set current transaction in Singapore (less than 1 hour later)
        df.loc[idx, 'Merchant_GPS_Lat'] = 1.3521
        df.loc[idx, 'Merchant_GPS_Lon'] = 103.8198
        # Make the time difference small (e.g., 20 minutes)
        df.loc[idx, 'Transaction_Time'] = df.loc[idx, 'Prev_Txn_Time'] + pd.Timedelta(minutes=20)
        df.loc[idx, 'Volume_Liters'] = np.random.uniform(40, 60) # High volume for smuggling

    # Anomaly 2: Excessive Volume (Large repetitive purchases) 
    for i in range(5):
        idx = np.random.randint(10, 950)
        df.loc[idx, 'Volume_Liters'] = np.random.uniform(80, 120) # Significantly higher volume

    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Distance and Implied Speed features."""
    logger.info("Engineering features (Distance/Speed)...")

    # 1. Calculate Distance_KM using Haversine (via geopy)
    def calculate_distance(row):
        if pd.isna(row['Prev_Txn_GPS_Lat']):
            return 0  # Cannot calculate distance for first transaction
        coords_prev = (row['Prev_Txn_GPS_Lat'], row['Prev_Txn_GPS_Lon'])
        coords_curr = (row['Merchant_GPS_Lat'], row['Merchant_GPS_Lon'])
        return geodesic(coords_prev, coords_curr).km

    df['Distance_KM'] = df.apply(calculate_distance, axis=1)

    # 2. Calculate Time Difference in Hours
    df['Time_Diff_Hours'] = (df['Transaction_Time'] - df['Prev_Txn_Time']).dt.total_seconds() / 3600
    df['Time_Diff_Hours'].fillna(1, inplace=True)

    # 3. Calculate Implied Speed KPH
    df['Implied_Speed_KPH'] = np.where(
        df['Time_Diff_Hours'] > 0,
        df['Distance_KM'] / df['Time_Diff_Hours'],
        # If time diff is 0 or less, set speed to a very high number to flag it
        Config.SPEED_THRESHOLD_KPH * 2
    )
    
    # 4. Create Anomaly Flag (for verification/testing only)
    df['Is_Anomaly'] = np.where(
        (df['Implied_Speed_KPH'] > Config.SPEED_THRESHOLD_KPH) | 
        (df['Volume_Liters'] > 70), 
        -1, # Anomaly
        1  # Normal
    )

    return df

def enhanced_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add more sophisticated features for better anomaly detection."""
    logger.info("Adding enhanced features...")
    
    # Start with basic features
    df = feature_engineer(df)
    
    # Additional features
    # 1. Time-based features
    df['Hour_of_Day'] = df['Transaction_Time'].dt.hour
    df['Day_of_Week'] = df['Transaction_Time'].dt.dayofweek
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
    
    # 2. Behavioral features
    df['Volume_Deviation'] = df.groupby('IC_No')['Volume_Liters'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    ).fillna(0)
    
    # 3. Frequency features
    df['Txn_Frequency'] = df.groupby('IC_No')['Transaction_Time'].diff().dt.total_seconds() / 3600
    df['Txn_Frequency'].fillna(df['Txn_Frequency'].mean(), inplace=True)
    
    # 4. Cumulative features
    df['Cumulative_Volume_Day'] = df.groupby(
        [df['IC_No'], df['Transaction_Time'].dt.date]
    )['Volume_Liters'].cumsum()
    
    return df

# --- 2. MODEL TRAINING & SAVING ---

def train_and_save_model(df: pd.DataFrame):
    """Trains the Isolation Forest model and saves it."""
    logger.info("Training Isolation Forest model...")

    # Features to use for Anomaly Detection
    FEATURES = [
        'Volume_Liters', 
        'Distance_KM', 
        'Implied_Speed_KPH',
        'Volume_Deviation',
        'Txn_Frequency',
        'Cumulative_Volume_Day'
    ]

    # Select the features for training
    X = df[FEATURES]

    # Initialize the Isolation Forest model
    model = IsolationForest(**Config.MODEL_PARAMS)

    # Train the model
    model.fit(X)
    
    # Store which features were used
    model.feature_names_in_ = FEATURES

    # Save the model with correct .pkl extension
    dump(model, Config.MODEL_PATH)
    logger.info(f"Model trained and saved to {Config.MODEL_PATH}")
    
    return model

# --- 3. FASTAPI SETUP WITH LIFESPAN MANAGER ---

# Define the data structure for the incoming transaction
class Transaction(BaseModel):
    ic_no: str
    transaction_time: datetime.datetime
    volume_liters: float
    merchant_gps_lat: float
    merchant_gps_lon: float
    # These represent the *last* known transaction details for the IC_No
    prev_txn_time: datetime.datetime
    prev_txn_gps_lat: float
    prev_txn_gps_lon: float

# Response model
class PredictionResponse(BaseModel):
    status: str
    ic_no: str
    prediction: str
    is_flagged: bool
    reasoning: str
    dashboard_alert: Optional[str] = None
    anomaly_score: Optional[float] = None
    features: Optional[Dict[str, float]] = None

# Global model variable
loaded_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI (replaces on_event)."""
    # Startup: Load model
    global loaded_model
    try:
        if os.path.exists(Config.MODEL_PATH):
            loaded_model = load(Config.MODEL_PATH)
            logger.info(f"✅ Model loaded successfully from {Config.MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model file {Config.MODEL_PATH} not found.")
            logger.info("To create a model, run: python anomaly_detection.py train")
            loaded_model = None
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        loaded_model = None
    
    yield  # App runs here
    
    # Shutdown: Cleanup if needed
    logger.info("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="PQC/TAD Anomaly Detection Backend",
    description="Transactional Anomaly Detection service for subsidy protection.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information."""
    model_status = "loaded" if loaded_model is not None else "not loaded"
    return {
        "message": "PQC/TAD Anomaly Detection API",
        "version": "1.0.0",
        "status": "operational",
        "model_status": model_status,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model_info",
            "predict": "/predict_anomaly",
            "train": "/train_model (POST)"
        },
        "instructions": {
            "train_model": "Run: python anomaly_detection.py train",
            "start_api": "Run: python anomaly_detection.py api",
            "test": "Run: python anomaly_detection.py test"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    model_loaded = loaded_model is not None
    status = "healthy" if model_loaded else "degraded"
    return {
        "status": status,
        "model_loaded": model_loaded,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_file_exists": os.path.exists(Config.MODEL_PATH)
    }

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    if loaded_model is None:
        return {
            "error": "Model not loaded",
            "instructions": "Train a model first using: python anomaly_detection.py train or POST /train_model"
        }
    
    return {
        "model_type": str(type(loaded_model)),
        "model_params": Config.MODEL_PARAMS,
        "features_used": getattr(loaded_model, 'feature_names_in_', ['Volume_Liters', 'Distance_KM', 'Implied_Speed_KPH']),
        "contamination": loaded_model.contamination,
        "n_estimators": loaded_model.n_estimators,
        "model_file": Config.MODEL_PATH,
        "model_file_size": f"{os.path.getsize(Config.MODEL_PATH) / 1024:.2f} KB" if os.path.exists(Config.MODEL_PATH) else "N/A"
    }

@app.post("/predict_anomaly", response_model=PredictionResponse)
async def predict_anomaly(txn: Transaction):
    """
    Receives a new transaction and returns a prediction (Normal or Suspicious).
    
    **Example request body:**
    ```json
    {
        "ic_no": "IC001",
        "transaction_time": "2025-01-01T12:00:00",
        "volume_liters": 45.5,
        "merchant_gps_lat": 3.0,
        "merchant_gps_lon": 101.5,
        "prev_txn_time": "2025-01-01T11:30:00",
        "prev_txn_gps_lat": 3.1,
        "prev_txn_gps_lon": 101.6
    }
    ```
    """
    try:
        # Validate input
        if txn.volume_liters <= 0:
            raise HTTPException(status_code=400, detail="Volume must be positive")
        
        if loaded_model is None:
            raise HTTPException(
                status_code=503, 
                detail="Prediction model not loaded. Please train a model first using: python anomaly_detection.py train or POST /train_model"
            )
        
        logger.info(f"Processing transaction for IC: {txn.ic_no}")
        
        # --- Feature Engineering for the new transaction ---
        
        # 1. Calculate Time Difference in Hours
        time_diff = (txn.transaction_time - txn.prev_txn_time).total_seconds() / 3600
        
        # 2. Calculate Distance_KM
        coords_prev = (txn.prev_txn_gps_lat, txn.prev_txn_gps_lon)
        coords_curr = (txn.merchant_gps_lat, txn.merchant_gps_lon)
        distance_km = geodesic(coords_prev, coords_curr).km
        
        # 3. Calculate Implied Speed KPH
        if time_diff > 0:
            implied_speed = distance_km / time_diff
        else:
            # Flag immediately if time difference is zero or negative
            implied_speed = Config.SPEED_THRESHOLD_KPH * 2
        
        # 4. Calculate additional features
        # For simplicity, using default values for behavioral features
        # In production, you'd fetch historical data to calculate these properly
        volume_deviation = 0  # Would need historical data
        txn_frequency = time_diff  # Using time diff as proxy
        cumulative_volume_day = txn.volume_liters  # Would need daily history
        
        # --- Prepare features for the model ---
        
        input_features = pd.DataFrame({
            'Volume_Liters': [txn.volume_liters],
            'Distance_KM': [distance_km],
            'Implied_Speed_KPH': [implied_speed],
            'Volume_Deviation': [volume_deviation],
            'Txn_Frequency': [txn_frequency],
            'Cumulative_Volume_Day': [cumulative_volume_day]
        })
        
        # Ensure we only use features that the model was trained on
        if hasattr(loaded_model, 'feature_names_in_'):
            # Reorder columns to match training data
            input_features = input_features[loaded_model.feature_names_in_]
        
        # Get Prediction and Anomaly Score
        # Model returns 1 for Normal, -1 for Anomaly
        prediction = loaded_model.predict(input_features)[0]
        anomaly_score = loaded_model.score_samples(input_features)[0]
        
        is_anomaly = prediction == -1
        
        # Determine reasoning
        reasons = []
        if implied_speed > Config.SPEED_THRESHOLD_KPH:
            reasons.append(f"High implied speed ({implied_speed:.2f} KPH)")
        if txn.volume_liters > 70:
            reasons.append(f"High volume ({txn.volume_liters:.2f} liters)")
        
        reasoning = " | ".join(reasons) if reasons else "Normal transaction pattern"
        
        # Create response
        response = {
            "status": "success",
            "ic_no": txn.ic_no,
            "prediction": "Suspicious" if is_anomaly else "Normal",
            "is_flagged": is_anomaly,
            "reasoning": reasoning,
            "dashboard_alert": "HIGH ANOMALY SCORE" if is_anomaly else None,
            "anomaly_score": float(anomaly_score),
            "features": {
                "volume_liters": float(txn.volume_liters),
                "distance_km": float(distance_km),
                "implied_speed_kph": float(implied_speed)
            }
        }
        
        logger.info(f"Prediction for {txn.ic_no}: {'🔴 SUSPICIOUS' if is_anomaly else '🟢 NORMAL'}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/train_model")
async def train_model_endpoint():
    """Endpoint to train a new model (for admin use)."""
    try:
        logger.info("Training new model via API endpoint...")
        df_raw = create_sample_data()
        df_features = enhanced_feature_engineer(df_raw)
        model = train_and_save_model(df_features)
        
        # Reload the model
        global loaded_model
        loaded_model = load(Config.MODEL_PATH)
        
        # Get some statistics
        anomaly_count = int((df_features['Is_Anomaly'] == -1).sum())
        total_count = len(df_features)
        
        return {
            "status": "success",
            "message": "Model trained and loaded successfully",
            "model_file": Config.MODEL_PATH,
            "samples_used": total_count,
            "anomalies_in_training": anomaly_count,
            "anomaly_percentage": f"{(anomaly_count/total_count*100):.2f}%",
            "instructions": "Model is now ready for predictions at /predict_anomaly"
        }
    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/check_model")
async def check_model():
    """Check if model exists and can be loaded."""
    model_exists = os.path.exists(Config.MODEL_PATH)
    model_loaded = loaded_model is not None
    
    if model_exists:
        file_size = os.path.getsize(Config.MODEL_PATH)
        file_size_mb = file_size / (1024 * 1024)
    else:
        file_size_mb = 0
    
    return {
        "model_exists": model_exists,
        "model_loaded": model_loaded,
        "model_file": Config.MODEL_PATH,
        "file_size_mb": round(file_size_mb, 3),
        "instructions": "Train model: python anomaly_detection.py train or POST /train_model"
    }

# --- EXECUTION ---
def main():
    """Main entry point with command line arguments."""
    print("\n" + "="*60)
    print("PQC/TAD ANOMALY DETECTION SYSTEM")
    print("="*60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            print("\n🚀 TRAINING MODE")
            print("-" * 40)
            
            # Check if model already exists
            if os.path.exists(Config.MODEL_PATH):
                print(f"⚠️ Model file {Config.MODEL_PATH} already exists.")
                response = input("Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("Training cancelled.")
                    return
            
            # Create and train model
            print("Creating sample data...")
            df_raw = create_sample_data()
            df_features = enhanced_feature_engineer(df_raw)
            
            # Show statistics
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total transactions: {len(df_features)}")
            print(f"   Unique IC numbers: {df_features['IC_No'].nunique()}")
            print(f"   Date range: {df_features['Transaction_Time'].min().date()} to {df_features['Transaction_Time'].max().date()}")
            
            anomaly_count = (df_features['Is_Anomaly'] == -1).sum()
            anomaly_percentage = (anomaly_count/len(df_features)*100)
            print(f"   Injected anomalies: {anomaly_count} ({anomaly_percentage:.1f}%)")
            
            # Show examples
            anomalies = df_features[df_features['Is_Anomaly'] == -1].head(3)
            if not anomalies.empty:
                print(f"\n🔍 Example anomalies detected:")
                for _, row in anomalies.iterrows():
                    speed_status = "🚨" if row['Implied_Speed_KPH'] > Config.SPEED_THRESHOLD_KPH else "⚠️"
                    volume_status = "🚨" if row['Volume_Liters'] > 70 else "⚠️"
                    print(f"   IC: {row['IC_No']:6s} | Speed: {row['Implied_Speed_KPH']:7.1f} KPH {speed_status} | Volume: {row['Volume_Liters']:6.1f}L {volume_status}")
            
            # Train model
            print("\n🤖 Training Isolation Forest model...")
            train_and_save_model(df_features)
            
            # Verify model was created
            if os.path.exists(Config.MODEL_PATH):
                file_size = os.path.getsize(Config.MODEL_PATH) / 1024
                print(f"\n✅ Model saved to {Config.MODEL_PATH} ({file_size:.1f} KB)")
                print("\nNext steps:")
                print("   1. Start API: python anomaly_detection.py api")
                print("   2. Test API: python anomaly_detection.py test")
                print("   3. Access docs: http://127.0.0.1:8000/docs")
            else:
                print(f"\n❌ ERROR: Model file was not created!")
                
        elif command == "api":
            print("\n🌐 API MODE")
            print("-" * 40)
            
            # Check if model exists
            if not os.path.exists(Config.MODEL_PATH):
                print(f"⚠️ Warning: Model file {Config.MODEL_PATH} not found!")
                print("\nTo train a model, run:")
                print("   python anomaly_detection.py train")
                print("\nStarting API anyway - you can train via POST /train_model")
            
            print(f"\nStarting API Server on {Config.API_HOST}:{Config.API_PORT}")
            print("\n📋 Available endpoints:")
            print(f"   📄 Documentation: http://{Config.API_HOST}:{Config.API_PORT}/docs")
            print(f"   ❤️  Health check: http://{Config.API_HOST}:{Config.API_PORT}/health")
            print(f"   🤖 Model info: http://{Config.API_HOST}:{Config.API_PORT}/model_info")
            print(f"   🔍 Predict: http://{Config.API_HOST}:{Config.API_PORT}/predict_anomaly")
            print(f"   🏋️  Train: http://{Config.API_HOST}:{Config.API_PORT}/train_model (POST)")
            print("\nPress CTRL+C to stop the server")
            print("-" * 40)
            
            uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
            
        elif command == "test":
            print("\n🧪 TEST MODE")
            print("-" * 40)
            
            # Check if model exists
            if not os.path.exists(Config.MODEL_PATH):
                print(f"❌ Model file {Config.MODEL_PATH} not found!")
                print("Train a model first: python anomaly_detection.py train")
                return
            
            print("Creating sample transaction data...")
            
            # Create test transactions
            now = datetime.datetime.now()
            
            # Normal transaction
            normal_txn = Transaction(
                ic_no="IC001",
                transaction_time=now,
                volume_liters=35.5,
                merchant_gps_lat=3.1390,  # Kuala Lumpur
                merchant_gps_lon=101.6869,
                prev_txn_time=now - datetime.timedelta(hours=2),
                prev_txn_gps_lat=3.0738,  # Nearby location
                prev_txn_gps_lon=101.5183
            )
            
            # Suspicious transaction (impossible travel)
            suspicious_txn = Transaction(
                ic_no="IC002",
                transaction_time=now,
                volume_liters=85.0,
                merchant_gps_lat=1.3521,  # Singapore
                merchant_gps_lon=103.8198,
                prev_txn_time=now - datetime.timedelta(minutes=30),
                prev_txn_gps_lat=1.4927,  # Johor, Malaysia
                prev_txn_gps_lon=103.7414
            )
            
            print("\n📝 Sample transactions created:")
            print("\n1. NORMAL Transaction:")
            print(f"   IC: {normal_txn.ic_no}")
            print(f"   Volume: {normal_txn.volume_liters}L")
            print(f"   Time between transactions: 2 hours")
            print(f"   Locations: Kuala Lumpur area")
            
            print("\n2. SUSPICIOUS Transaction (Impossible Travel):")
            print(f"   IC: {suspicious_txn.ic_no}")
            print(f"   Volume: {suspicious_txn.volume_liters}L (⚠️ High)")
            print(f"   Time between transactions: 30 minutes")
            print(f"   Locations: Johor, Malaysia → Singapore")
            print(f"   Distance: ~30km")
            print(f"   Implied speed: ~60 KPH (plausible but high volume)")
            
            print("\n📋 To test the API:")
            print("1. Start API: python anomaly_detection.py api")
            print("2. Use curl or Postman to send POST requests to /predict_anomaly")
            print("\nExample curl command for suspicious transaction:")
            print(f"""curl -X POST "http://127.0.0.1:8000/predict_anomaly" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "ic_no": "IC002",
    "transaction_time": "{suspicious_txn.transaction_time.isoformat()}",
    "volume_liters": {suspicious_txn.volume_liters},
    "merchant_gps_lat": {suspicious_txn.merchant_gps_lat},
    "merchant_gps_lon": {suspicious_txn.merchant_gps_lon},
    "prev_txn_time": "{suspicious_txn.prev_txn_time.isoformat()}",
    "prev_txn_gps_lat": {suspicious_txn.prev_txn_gps_lat},
    "prev_txn_gps_lon": {suspicious_txn.prev_txn_gps_lon}
  }}'""")
            
        elif command == "help" or command == "--help" or command == "-h":
            print("\n📚 HELP MENU")
            print("-" * 40)
            print("\nAvailable commands:")
            print("  train    - Train and save a new anomaly detection model")
            print("  api      - Start the FastAPI server")
            print("  test     - Create sample test data and show usage")
            print("  help     - Show this help menu")
            
            print("\nUsage:")
            print("  python anomaly_detection.py train     # Train model first")
            print("  python anomaly_detection.py api       # Start API server")
            print("  python anomaly_detection.py test      # Test with sample data")
            
            print("\nAPI Endpoints (when server is running):")
            print("  GET  /           - API information")
            print("  GET  /health     - Health check")
            print("  GET  /model_info - Model information")
            print("  POST /predict_anomaly - Make predictions")
            print("  POST /train_model - Train new model via API")
            
        else:
            print(f"\n❌ Unknown command: {command}")
            print("Use: python anomaly_detection.py [train|api|test|help]")
            
    else:
        # No command specified - show help
        print("\nNo command specified.")
        print("\nUsage: python anomaly_detection.py [train|api|test|help]")
        print("\nQuick start:")
        print("  1. python anomaly_detection.py train    # Train the model")
        print("  2. python anomaly_detection.py api      # Start the API server")
        print("  3. python anomaly_detection.py test     # Test with sample data")
        print("\nFor more details: python anomaly_detection.py help")

if __name__ == '__main__':
    main()
