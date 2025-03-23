from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, Dealer, Shopkeeper
from pydantic import BaseModel, EmailStr
from bcrypt import hashpw, gensalt, checkpw
from models.demand_model import DemandForecastModel
import pandas as pd
from utils.data_preprocessing import preprocess_data
import traceback

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
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

# Login Endpoints
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

# Signup Endpoints
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

# Inventory Prediction Endpoint
@app.post("/inventory/predict")
async def predict_inventory(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = await file.read()
        df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
        X, y, scaler = preprocess_data(df)
        model = DemandForecastModel()
        model.train(X, y)  # Train the model with uploaded data
        predictions = model.predict(X)
        # Return a single averaged prediction for simplicity
        avg_prediction = predictions.mean() if predictions.size > 0 else 0
        print(f"Generated prediction: {avg_prediction}")
        return {"predictions": [avg_prediction]}  # Single value for the shop
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()  # Print full stack trace for debugging
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")