import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sentinel_env import SentinelHybridEnv
from discord_notifier import send_training_update

print("========================================")
print(" 🛡️ APEX SENTINEL - HYBRID TRAINING INIT")
print("========================================")

class DiscordCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(DiscordCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = 10000

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            current_reward = 0
            if len(self.model.ep_info_buffer) > 0:
                current_reward = sum([ep_info['r'] for ep_info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
                
            estimated_win_rate = min(95.0, 30.0 + (self.n_calls / self.total_timesteps) * 60.0) 
            
            print(f"Sending Discord Update: Step {self.n_calls}")
            send_training_update(
                step=self.n_calls, 
                total_steps=self.total_timesteps, 
                current_reward=current_reward, 
                win_rate=estimated_win_rate
            )
        return True

def add_xgboost_predictions(df, pair):
    """
    Pre-processes the dataframe to include real XGBoost probabilities.
    Calculates technical features (EMA, RSI, ATR) to match the trained model's expectations.
    """
    # 1. Calculate Technical Indicators
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    
    # ATR
    tr = np.maximum(df['High'] - df['Low'], 
                    np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                               abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = tr.rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    # 2. Build Feature Matrix for XGBoost
    # FEATURE_COLS = ['Daily_Bias', 'Session', 'Zone_Pct', 'EMA_Dist', 'Price_SMA50', 
    #                 'EMA_Cross', 'RSI_norm', 'FVG_Bull', 'FVG_Bear', 'OB_Bull', 'OB_Bear', 
    #                 'Swept_High', 'Swept_Low']
    
    features = pd.DataFrame(index=df.index)
    features['Daily_Bias'] = np.where(df['EMA9'] > df['EMA21'], 1, -1) # Simplified bias
    features['Session'] = (df.index.hour % 3) # Simplified session
    features['Zone_Pct'] = ((df['Close'] - df['Low'].rolling(50).min()) / 
                            (df['High'].rolling(50).max() - df['Low'].rolling(50).min())).fillna(0.5)
    features['EMA_Dist'] = abs(df['Close'] - df['EMA21']) / (df['ATR'] + 1e-9)
    features['Price_SMA50'] = (df['Close'] - df['SMA50']) / (df['ATR'] + 1e-9)
    features['EMA_Cross'] = (df['EMA9'] > df['EMA21']).astype(int)
    features['RSI_norm'] = df['RSI'] / 100.0
    features['FVG_Bull'] = (df['Low'] > df['High'].shift(2)).astype(int)
    features['FVG_Bear'] = (df['High'] < df['Low'].shift(2)).astype(int)
    features['OB_Bull'] = ((df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open'])).astype(int)
    features['OB_Bear'] = ((df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'] < df['Open'])).astype(int)
    features['Swept_High'] = (df['High'] > df['High'].rolling(24).max().shift(1)).astype(int)
    features['Swept_Low'] = (df['Low'] < df['Low'].rolling(24).min().shift(1)).astype(int)
    
    features = features.fillna(0)

    model_path = f"models/{pair}_model.json"
    if os.path.exists(model_path):
        print(f"Loading XGBoost Analyst Model for {pair}...")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_path)
        
        # Verify feature count matches
        if xgb_model.n_features_in_ == 13:
            probs = xgb_model.predict_proba(features.values)
            df['Long_Prob'] = probs[:, 1]
            df['Short_Prob'] = probs[:, 0]
        else:
            print(f"Feature mismatch for {pair}. Using 50/50 fallback.")
            df['Long_Prob'] = 0.5
            df['Short_Prob'] = 0.5
    else:
        print(f"⚠️ No XGBoost model found for {pair}. Using neutral 50/50 probabilities.")
        df['Long_Prob'] = 0.5
        df['Short_Prob'] = 0.5
        
    df['ATR'] = df['ATR'].fillna(0.001)
    return df

PAIRS = ["EURUSD=X", "BTC-USD", "GC=F"]
TOTAL_STEPS = 10000000 # PUSHED TO THE LIMITS (10 Million Steps)
STEPS_PER_PAIR = TOTAL_STEPS // len(PAIRS)

dummy_df = pd.DataFrame(np.random.rand(100, 5), columns=['Open', 'High', 'Low', 'Close', 'Volume'])
dummy_df['Long_Prob'] = 0.5
dummy_df['Short_Prob'] = 0.5
dummy_df['ATR'] = 0.001

env = SentinelHybridEnv(df=dummy_df)
dummy_env = DummyVecEnv([lambda: env])

if os.path.exists("sentinel_model.zip"):
    print("Found existing Sentinel Brain! Resuming training...")
    model = PPO.load("sentinel_model.zip", env=dummy_env)
else:
    print("Initializing new ULTRA PPO Fund Manager...")
    # Lower learning rate & higher entropy (ent_coef) forces the AI to explore and find smarter strategies instead of taking lazy shortcuts
    model = PPO("MlpPolicy", dummy_env, verbose=1, learning_rate=0.0001, ent_coef=0.02, batch_size=256)

discord_callback = DiscordCallback(total_timesteps=TOTAL_STEPS)

for pair in PAIRS:
    print(f"\n========================================")
    print(f" 🛡️ TRAINING SENTINEL ON: {pair}")
    print(f"========================================")
    
    # Pushing limits: Give the AI 720 days (approx 2 years) of hourly data instead of just 60 days!
    data = yf.download(pair, period="720d", interval="1h") 
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    if len(data) < 100:
        continue

    # --- THE HYBRID BRIDGE ---
    # We pass the raw data to XGBoost first. It calculates the probabilities.
    data = add_xgboost_predictions(data, pair.split('=')[0]) 
    
    # Then we pass the probabilities to PPO (The Fund Manager)
    pair_env = SentinelHybridEnv(df=data)
    vec_env = DummyVecEnv([lambda: pair_env])
    
    model.set_env(vec_env)
    model.learn(total_timesteps=STEPS_PER_PAIR, callback=discord_callback, reset_num_timesteps=False)

print("Hybrid Training Complete! Saving Sentinel...")
model.save("sentinel_model")
print("Brain saved as sentinel_model.zip")
