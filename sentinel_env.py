import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class SentinelHybridEnv(gym.Env):
    """
    Apex Sentinel Sandbox
    Hybrid Engine: Learns how to manage risk based on XGBoost predictions.
    """
    def __init__(self, df, initial_balance=10000):
        super(SentinelHybridEnv, self).__init__()
        
        # The dataframe must contain 'Close', 'Long_Prob', 'Short_Prob', 'ATR'
        self.df = df.reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.spread_fee = 0.00015 # 0.015% percentage spread
        
        # ACTION SPACE: Position Sizing & Risk Management
        # 0: Hold Flat
        # 1: Buy Light (1% Risk), 2: Buy Normal (5% Risk), 3: Buy MAX (10% Risk)
        # 4: Sell Light (1% Risk), 5: Sell Normal (5% Risk), 6: Sell MAX (10% Risk)
        # 7: Close Position
        self.action_space = spaces.Discrete(8)
        
        # OBSERVATION SPACE: PPO sees the Analyst's probabilities, not raw price!
        # [Long_Prob, Short_Prob, ATR, Unrealized_PnL_Pct, Current_Position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0 # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.trade_duration = 0
        self.lot_size_multiplier = 0
        self.done = False
        return self._get_obs(), {}
        
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        
        # Calculate unrealized PnL percentage if in a trade
        unrealized_pnl_pct = 0
        if self.position != 0:
            current_price = row['Close']
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - current_price) / self.entry_price
            unrealized_pnl_pct *= 100.0 # Leverage multiplier for scaling
            
        obs = np.array([
            row.get('Long_Prob', 0.5),
            row.get('Short_Prob', 0.5),
            row.get('ATR', 0.001),
            unrealized_pnl_pct,
            self.position
        ], dtype=np.float32)
        
        return obs
        
    def step(self, action):
        row = self.df.iloc[self.current_step]
        current_price = row['Close']
        reward = 0
        
        if action in [1, 2, 3] and self.position == 0:
            self.position = 1
            self.entry_price = current_price * (1 + self.spread_fee)
            self.trade_duration = 0
            self.lot_size_multiplier = [0.01, 0.05, 0.10][action - 1]
            
        elif action in [4, 5, 6] and self.position == 0:
            self.position = -1
            self.entry_price = current_price * (1 - self.spread_fee)
            self.trade_duration = 0
            self.lot_size_multiplier = [0.01, 0.05, 0.10][action - 4]
            
        elif self.position != 0 and action == 0:
            self.trade_duration += 1
            price_change_pct = (current_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - current_price) / self.entry_price
            
            # Heavy penalty for riding massive drawdowns (forces it to cut losses!)
            if price_change_pct < -0.01: # 1% raw move against us
                reward -= abs(price_change_pct) * self.lot_size_multiplier * 100.0 * self.balance
                
        elif action == 7 and self.position != 0:
            price_change_pct = (current_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - current_price) / self.entry_price
            position_size = self.balance * self.lot_size_multiplier
            actual_profit = position_size * price_change_pct * 100.0 # 100x Leverage
            
            reward += actual_profit
            self.balance += actual_profit
            
            self.position = 0
            self.entry_price = 0
            self.lot_size_multiplier = 0
            
        self.current_step += 1
        if self.current_step >= len(self.df) - 1 or self.balance <= 0:
            self.done = True
            
        return self._get_obs(), reward, self.done, False, {}
