import requests
import json
import os
from datetime import datetime

# The Webhook URL you provided
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1553775394887303240/QY3e3vfF1qzSCtKdvmpyU69LVUy_yajqcR1uSBjb2kfeYeYoltlvpBxe1iF9aAEQfHLP")

def send_training_update(step, total_steps, current_reward, win_rate):
    """ Sends Kaggle/Colab Training Progress to Discord """
    progress = (step / total_steps) * 100
    embed = {
        "title": "🧠 Project Apex: Training Update",
        "description": f"The AI is actively learning in the Sandbox. \n**Progress:** {progress:.2f}%",
        "color": 3447003, # Blue
        "fields": [
            {"name": "Current Step", "value": f"{step:,} / {total_steps:,}", "inline": True},
            {"name": "Avg Reward", "value": f"${current_reward:,.2f}", "inline": True},
            {"name": "Est. Win Rate", "value": f"{win_rate:.1f}%", "inline": True}
        ],
        "footer": {"text": "Apex Primal Engine • Deep Reinforcement Learning"}
    }
    _send_webhook(embed)


def send_live_signal(pair, action, trade_type, grade, entry, sl, tp, rrr, win_rate):
    """ Sends a Live Trading Signal to Discord when the AI finds a setup """
    
    # Color coding based on Grade
    color_map = {"S-Class": 16711680, "A++": 16753920, "A": 65280, "B": 255, "C": 8421504, "D": 0}
    color = color_map.get(grade, 3447003)
    
    # Emoji formatting
    action_emoji = "🟢 BUY (LONG)" if action.upper() == "BUY" else "🔴 SELL (SHORT)"
    
    embed = {
        "title": f"🚨 APEX SIGNAL DETECTED: {pair}",
        "description": f"**{action_emoji}**\n\n**Setup Grade:** `{grade}`\n**Trade Type:** `{trade_type}`",
        "color": color,
        "fields": [
            {"name": "Entry Price", "value": f"{entry:.5f}", "inline": True},
            {"name": "Stop Loss (SL)", "value": f"{sl:.5f}", "inline": True},
            {"name": "Take Profit (TP)", "value": f"{tp:.5f}", "inline": True},
            {"name": "Risk/Reward (RRR)", "value": f"1:{rrr:.1f}", "inline": True},
            {"name": "AI Confidence (WR)", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
        ],
        "footer": {"text": "Apex Primal Engine • Live Market Scan"}
    }
    _send_webhook(embed)


def send_trade_result(pair, result_type, profit_pips):
    """ Sends an update when a trade hits TP or SL """
    
    if result_type.upper() == "TP":
        title = f"✅ TAKE PROFIT HIT: {pair}"
        color = 65280 # Green
        desc = f"The AI successfully closed the trade in profit.\n**Secured:** `+{profit_pips} pips`"
    else:
        title = f"❌ STOP LOSS HIT: {pair}"
        color = 16711680 # Red
        desc = f"The AI cut its losses to protect the account.\n**Lost:** `{profit_pips} pips`"
        
    embed = {
        "title": title,
        "description": desc,
        "color": color,
        "timestamp": datetime.utcnow().isoformat()
    }
    _send_webhook(embed)


def _send_webhook(embed):
    payload = {"embeds": [embed]}
    try:
        response = requests.post(WEBHOOK_URL, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        if response.status_code != 204:
            print(f"Failed to send Discord message: {response.status_code}")
    except Exception as e:
        print(f"Error sending Discord update: {e}")
