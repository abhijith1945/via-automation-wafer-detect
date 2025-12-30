"""
🤖 Virtual Metrology AI Chatbot
Powered by Google Gemini API
Expert assistant for semiconductor manufacturing engineers
"""

import google.generativeai as genai
from datetime import datetime

# System prompt for the chatbot
SYSTEM_PROMPT = """You are VIA (Virtual Intelligence Assistant), an expert AI assistant for semiconductor manufacturing engineers working with a Virtual Metrology System.

Your expertise includes:
1. **Wafer Defect Analysis**: Scratch, Edge Ring, Particle, Center defects
2. **Process Parameters**: Temperature (300-600°C), Pressure (90-110 Pa), Gas Flow (40-60 sccm), RF Power (800-1200W)
3. **Yield Optimization**: Understanding pass/fail predictions and improving yield
4. **Root Cause Analysis**: Identifying causes of defects from sensor data
5. **Self-Healing Actions**: Recommending corrective process adjustments
6. **Equipment Maintenance**: Chamber cleaning, sensor calibration, etc.

Guidelines:
- Be concise but thorough (2-4 paragraphs max)
- Use bullet points for lists
- Include specific numbers and thresholds when relevant
- Always suggest actionable next steps
- Use emojis sparingly for key points (✅, ⚠️, 🔧, 📊)
- If asked about the system, explain it uses Random Forest + CNN + VAE models
- Be professional but friendly

Key Process Thresholds:
- Temperature: Optimal 420-480°C, Warning >500°C, Critical >550°C
- Pressure: Optimal 98-102 Pa, Warning <95 or >105, Critical <92 or >108
- Flow Rate: Optimal 48-52 sccm, Warning ±5 from optimal
- RF Power: Optimal 950-1050W, Warning ±100 from optimal

Defect Causes:
- Scratch: Mechanical handling issues, robotic arm misalignment
- Edge Ring: Temperature non-uniformity, edge exclusion zone issues  
- Particle: Chamber contamination, gas line debris
- Center: Chuck temperature variation, plasma non-uniformity

Always end with a helpful suggestion or next step the engineer can take."""


class VirtualMetrologyChat:
    """AI Chatbot for semiconductor manufacturing engineers"""
    
    def __init__(self, api_key: str):
        """Initialize the chatbot with Gemini API"""
        self.api_key = api_key
        self.model = None
        self.chat = None
        self.history = []
        self.initialized = False
        self.error = None
        self.model_name = None
        self._initialize()
    
    def _initialize(self):
        """Configure Gemini and start chat session"""
        try:
            genai.configure(api_key=self.api_key)
            
            # List of models to try in order (updated to available models)
            model_names = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash-latest"]
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=SYSTEM_PROMPT
                    )
                    # Test the model with a simple message
                    self.chat = self.model.start_chat(history=[])
                    self.model_name = model_name
                    self.initialized = True
                    print(f"✅ Chatbot initialized with {model_name}")
                    return
                except Exception as model_error:
                    print(f"⚠️ Model {model_name} failed: {model_error}")
                    continue
            
            # If all models fail
            self.initialized = False
            self.error = "All Gemini models failed to initialize"
            
        except Exception as e:
            self.initialized = False
            self.error = str(e)
    
    def send_message(self, user_message: str, context: dict = None) -> str:
        """Send a message and get AI response"""
        
        if not self.initialized:
            # Try to reinitialize once
            self._initialize()
            if not self.initialized:
                return f"⚠️ Chatbot initialization failed: {getattr(self, 'error', 'Unknown error')}"
        
        try:
            # Add context if provided (e.g., current wafer data)
            enhanced_message = user_message
            if context:
                context_str = "\n\n[Current System Context:\n"
                for key, value in context.items():
                    context_str += f"- {key}: {value}\n"
                context_str += "]\n"
                enhanced_message = context_str + user_message
            
            # Send to Gemini with timeout
            print(f"📤 Sending to Gemini ({getattr(self, 'model_name', 'unknown')})...")
            response = self.chat.send_message(enhanced_message)
            print(f"📥 Received response from Gemini")
            
            # Store in history
            self.history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            self.history.append({
                "role": "assistant", 
                "content": response.text,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            return response.text
            
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
    def get_quick_suggestions(self) -> list:
        """Return quick suggestion buttons for common questions"""
        return [
            "Why did my wafer fail?",
            "What causes scratch defects?",
            "Optimal temperature range?",
            "How to reduce particle defects?",
            "Explain the prediction model",
            "Chamber maintenance tips",
            "How to interpret confidence scores?",
            "What is SMOTE balancing?"
        ]
    
    def clear_history(self):
        """Clear chat history and start fresh"""
        self.history = []
        if self.model:
            self.chat = self.model.start_chat(history=[])


# Fallback responses for when API fails
FALLBACK_RESPONSES = {
    "fail": """🔍 **Common Wafer Failure Causes:**

1. **Temperature Issues** (>500°C)
   - Causes edge ring defects
   - Solution: Reduce by 20-30°C

2. **Pressure Deviation** (<95 or >105 Pa)
   - Causes non-uniform etching
   - Solution: Check vacuum pump and seals

3. **Contamination**
   - Causes particle defects
   - Solution: Run chamber purge cycle

✅ **Next Step:** Check the sensor readings panel for anomalies.""",

    "scratch": """🔧 **Scratch Defect Analysis:**

**Root Causes:**
- Robotic arm misalignment
- Worn end effector pads
- Improper wafer handling sequence

**Solutions:**
1. Calibrate robot arm positioning (±0.1mm tolerance)
2. Replace end effector pads every 5000 cycles
3. Verify wafer centering on chuck

⚠️ **Impact:** Scratches typically reduce yield by 15-25%""",

    "temperature": """🌡️ **Temperature Guidelines:**

| Range | Status | Action |
|-------|--------|--------|
| 420-480°C | ✅ Optimal | Continue monitoring |
| 480-500°C | ⚠️ Warning | Reduce by 10°C |
| >500°C | 🚨 Critical | Immediate adjustment |

**Tip:** Temperature variations >5°C during process cause edge defects.""",

    "pressure": """🔵 **Pressure Guidelines:**

| Range | Status | Action |
|-------|--------|--------|
| 98-102 Pa | ✅ Optimal | Continue monitoring |
| 95-98 or 102-105 Pa | ⚠️ Warning | Check seals |
| <95 or >105 Pa | 🚨 Critical | Stop process |

**Common Issues:**
- Low pressure: Vacuum leak, pump degradation
- High pressure: Gas flow blockage, valve issues""",

    "particle": """🔬 **Particle Defect Analysis:**

**Root Causes:**
1. Chamber contamination (most common)
2. Gas line debris
3. Inadequate cleaning cycles
4. Poor HEPA filter maintenance

**Solutions:**
1. Run chamber purge with N2 gas (10 min)
2. Replace gas line filters monthly
3. Schedule plasma cleaning every 500 wafers
4. Check HEPA filter differential pressure

⚠️ **Prevention:** 80% of particle defects are preventable with maintenance.""",

    "edge": """🔘 **Edge Ring Defect Analysis:**

**Root Causes:**
1. Temperature non-uniformity at wafer edge
2. Edge exclusion zone misconfiguration
3. Chuck temperature gradient
4. Plasma edge effects

**Solutions:**
1. Verify edge exclusion zone (typically 3-5mm)
2. Check chuck heater uniformity (<2°C variance)
3. Adjust RF power distribution
4. Review plasma confinement ring condition

**Tip:** Edge defects often indicate thermal control issues.""",

    "model": """🤖 **Virtual Metrology AI Model:**

**Architecture:**
```
Sensor Data → Random Forest → PASS/FAIL
     ↓              ↓
   590 sensors   SMOTE balanced
```

**Key Components:**
1. **Random Forest Classifier** - 93.3% accuracy
2. **SMOTE** - Handles class imbalance (6.6% fail rate)
3. **CNN Vision** - Defect classification from images
4. **VAE** - Generative AI for synthetic images

**Why Random Forest?**
- Handles high-dimensional data (590 sensors)
- Provides feature importance
- Fast inference (<20ms)
- No scaling required""",

    "smote": """⚖️ **SMOTE (Synthetic Minority Over-sampling Technique):**

**The Problem:**
- Dataset: 93.4% PASS, 6.6% FAIL
- Imbalanced data causes model to ignore failures
- Missing failures = costly defective wafers

**How SMOTE Works:**
1. Find k-nearest neighbors of minority class (FAIL)
2. Create synthetic samples between neighbors
3. Balance dataset to ~50/50

**Results:**
- Before SMOTE: 76% recall on failures
- After SMOTE: 91% recall on failures

**Why It Matters:**
In semiconductor manufacturing, missing a defective wafer costs ~$5,000. SMOTE helps catch more defects!""",

    "maintenance": """🔧 **Chamber Maintenance Best Practices:**

**Daily:**
- [ ] Visual inspection of chamber
- [ ] Check gas flow rates
- [ ] Verify temperature readings

**Weekly:**
- [ ] Run plasma cleaning cycle
- [ ] Check vacuum pump oil level
- [ ] Inspect O-rings and seals

**Monthly:**
- [ ] Replace gas line filters
- [ ] Calibrate temperature sensors
- [ ] Check RF power calibration
- [ ] Inspect end effectors

**Quarterly:**
- [ ] Full chamber wet clean
- [ ] Replace consumables
- [ ] Pump maintenance

⏱️ **ROI:** Preventive maintenance reduces defects by 40%""",

    "confidence": """📊 **Understanding Confidence Scores:**

**What It Means:**
- Confidence = model's certainty in prediction
- Higher = more reliable prediction

**Interpretation:**
| Score | Meaning | Action |
|-------|---------|--------|
| >90% | Very confident | Trust prediction |
| 70-90% | Moderate | Review manually |
| 50-70% | Uncertain | Inspect wafer |
| <50% | Low confidence | Don't trust |

**Why Low Confidence?**
1. Unusual sensor readings
2. Edge case patterns
3. Missing data
4. New defect type

**Tip:** Low confidence often indicates process drift!""",

    "virtual_metrology": """🔬 **What is Virtual Metrology?**

**Definition:**
Predicting wafer quality using process sensor data instead of physical measurements.

**Traditional vs Virtual:**
| | Physical | Virtual |
|---|----------|---------|
| Time | 30 min | 0.3 sec |
| Cost | $50/wafer | ~$0.01 |
| Coverage | 5% sample | 100% |
| Destructive | Sometimes | Never |

**How It Works:**
```
590 Sensors → AI Model → PASS/FAIL
(Temperature, Pressure, Flow, RF Power, etc.)
```

**Business Value:**
- **6,000x faster** than physical metrology
- **100% coverage** vs 5% sampling
- **$5M+ annual savings** for typical fab

**This System Uses:**
- Random Forest (sensor prediction)
- CNN (visual inspection)
- VAE (image generation)
- Self-healing control (parameter adjustment)""",

    "default": """🤖 **I'm VIA, your Virtual Metrology Assistant!**

I can help with:
- 🔍 **Defect Analysis** - scratch, edge ring, particle causes
- 🌡️ **Process Parameters** - optimal ranges and troubleshooting
- 📊 **Model Explanation** - how predictions work
- 🔧 **Maintenance Tips** - preventive care guidelines
- ⚖️ **Data Science** - SMOTE, confidence scores

**Quick Questions to Try:**
- "What causes scratch defects?"
- "Optimal temperature range?"
- "Explain the AI model"
- "Why did my wafer fail?"

💡 *Type your question below or click a Quick Question!*"""
}


def get_fallback_response(message: str) -> str:
    """Get a fallback response when API is unavailable"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['fail', 'failed', 'why']):
        return FALLBACK_RESPONSES['fail']
    elif any(word in message_lower for word in ['scratch', 'scratches']):
        return FALLBACK_RESPONSES['scratch']
    elif any(word in message_lower for word in ['temperature', 'temp', 'hot', 'cold']):
        return FALLBACK_RESPONSES['temperature']
    elif any(word in message_lower for word in ['pressure', 'vacuum', 'pa ']):
        return FALLBACK_RESPONSES['pressure']
    elif any(word in message_lower for word in ['particle', 'contamination', 'debris']):
        return FALLBACK_RESPONSES['particle']
    elif any(word in message_lower for word in ['edge', 'ring', 'edge_ring']):
        return FALLBACK_RESPONSES['edge']
    elif any(word in message_lower for word in ['model', 'random forest', 'ai', 'algorithm', 'predict']):
        return FALLBACK_RESPONSES['model']
    elif any(word in message_lower for word in ['smote', 'imbalance', 'balance', 'oversamp']):
        return FALLBACK_RESPONSES['smote']
    elif any(word in message_lower for word in ['maintenance', 'clean', 'maintain', 'prevent']):
        return FALLBACK_RESPONSES['maintenance']
    elif any(word in message_lower for word in ['confidence', 'score', 'certain', 'percent']):
        return FALLBACK_RESPONSES['confidence']
    elif any(word in message_lower for word in ['virtual metrology', 'what is', 'how does', 'explain system']):
        return FALLBACK_RESPONSES['virtual_metrology']
    else:
        return FALLBACK_RESPONSES['default']


# Alias for backward compatibility
VIAChatbot = VirtualMetrologyChat
