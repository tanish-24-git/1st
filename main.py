from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, Dealer, Shopkeeper
from pydantic import BaseModel, EmailStr
from bcrypt import hashpw, gensalt, checkpw
from models.demand_model import DemandForecastModel
import pandas as pd
import io
from utils.data_preprocessing import preprocess_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Login(BaseModel):
    email: EmailStr
    password: str

class DealerSignup(BaseModel):
    name: str
    email: EmailStr
    company_name: str
    location_name: str
    latitude: float
    longitude: float
    password: str

class ShopkeeperSignup(BaseModel):
    name: str
    email: EmailStr
    shop_name: str
    location_name: str
    latitude: float
    longitude: float
    domain: str
    password: str

@app.post("/dealer/login")
async def dealer_login(login: Login, db: Session = Depends(get_db)):
    dealer = db.query(Dealer).filter(Dealer.email == login.email).first()
    if not dealer or not checkpw(login.password.encode('utf-8'), dealer.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "dealer_id": dealer.dealer_id}

@app.post("/shopkeeper/login")
async def shopkeeper_login(login: Login, db: Session = Depends(get_db)):
    shopkeeper = db.query(Shopkeeper).filter(Shopkeeper.email == login.email).first()
    if not shopkeeper or not checkpw(login.password.encode('utf-8'), shopkeeper.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "shopkeeper_id": shopkeeper.shopkeeper_id}

@app.post("/dealer/signup")
async def dealer_signup(signup: DealerSignup, db: Session = Depends(get_db)):
    if db.query(Dealer).filter(Dealer.email == signup.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    password_hash = hashpw(signup.password.encode('utf-8'), gensalt()).decode('utf-8')
    new_dealer = Dealer(**signup.dict(exclude={"password"}), password_hash=password_hash)
    db.add(new_dealer)
    db.commit()
    db.refresh(new_dealer)
    return {"message": "Dealer signup successful", "dealer_id": new_dealer.dealer_id}

@app.post("/shopkeeper/signup")
async def shopkeeper_signup(signup: ShopkeeperSignup, db: Session = Depends(get_db)):
    if db.query(Shopkeeper).filter(Shopkeeper.email == signup.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    password_hash = hashpw(signup.password.encode('utf-8'), gensalt()).decode('utf-8')
    new_shopkeeper = Shopkeeper(**signup.dict(exclude={"password"}), password_hash=password_hash)
    db.add(new_shopkeeper)
    db.commit()
    db.refresh(new_shopkeeper)
    return {"message": "Shopkeeper signup successful", "shopkeeper_id": new_shopkeeper.shopkeeper_id}

@app.post("/inventory/predict")
async def predict_inventory(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        if not file.filename.endswith('.csv'):
            raise ValueError("File must be a CSV")
        contents = await file.read()
        if not contents:
            raise ValueError("File is empty")
        
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        if df.empty:
            raise ValueError("CSV file contains no data")
        
        logger.info(f"CSV loaded with {len(df)} rows and columns: {list(df.columns)}")
        X, y, scaler = preprocess_data(df)
        model = DemandForecastModel()
        model.train(X, y)
        predictions = model.predict(X)
        
        # Prepare detailed prediction results
        product_ids = df['Product_ID'].tolist()
        prediction_results = [
            {
                "product_id": str(pid),
                "product_name": f"Product {pid}",
                "predicted_demand": float(pred)
            }
            for pid, pred in zip(product_ids, predictions)
        ]
        
        # Sort by predicted_demand and take top 5
        top_5_predictions = sorted(prediction_results, key=lambda x: x["predicted_demand"], reverse=True)[:5]
        
        logger.info(f"Top 5 predictions: {top_5_predictions}")
        
        return {
            "success": True,
            "predictions": top_5_predictions,
            "message": "Top 5 predictions generated"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")