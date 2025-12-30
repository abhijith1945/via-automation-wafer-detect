"""
🤖 LLM-Powered Defect Report Generator
Generates human-readable explanations using template-based NLG
"""

import random
from datetime import datetime

class DefectReportGenerator:
    """
    Generative AI for Natural Language Defect Reports
    Uses template-based generation with variation
    """
    
    def __init__(self):
        self.templates = {
            'scratch': {
                'causes': [
                    "mechanical contact during wafer handling",
                    "debris on the chuck surface",
                    "worn robotic end-effector pads",
                    "particle contamination in the carrier",
                    "improper cassette alignment"
                ],
                'impacts': [
                    "may cause circuit discontinuity in affected die",
                    "could lead to yield loss in the scratch path",
                    "increases risk of particle generation in subsequent steps",
                    "may propagate during CMP processing"
                ],
                'actions': [
                    "Inspect and clean robotic handling equipment",
                    "Replace end-effector pads if wear detected",
                    "Verify cassette alignment sensors",
                    "Check for particles in FOUP environment",
                    "Review handling recipe parameters"
                ]
            },
            'edge_ring': {
                'causes': [
                    "non-uniform plasma distribution at wafer edge",
                    "improper ESC clamping force",
                    "edge exclusion zone contamination",
                    "temperature gradient near wafer periphery",
                    "gas flow non-uniformity"
                ],
                'impacts': [
                    "reduces usable die count at wafer edge",
                    "indicates potential chamber drift",
                    "may worsen over time without correction",
                    "affects edge die electrical parameters"
                ],
                'actions': [
                    "Recalibrate ESC clamping pressure",
                    "Adjust edge ring component position",
                    "Verify showerhead gas distribution",
                    "Check for edge deposition buildup",
                    "Review thermal control settings"
                ]
            },
            'particle': {
                'causes': [
                    "chamber component degradation",
                    "process byproduct accumulation",
                    "vacuum leak introducing contaminants",
                    "inadequate chamber cleaning cycle",
                    "gas line contamination"
                ],
                'impacts': [
                    "causes localized defects affecting multiple die",
                    "may indicate systemic chamber issue",
                    "risk of particle shower affecting lot",
                    "potential for killer defects"
                ],
                'actions': [
                    "Perform full chamber wet clean",
                    "Inspect and replace degraded components",
                    "Verify vacuum integrity with leak check",
                    "Increase purge cycle duration",
                    "Run particle qualification wafer"
                ]
            }
        }
        
        self.severity_adjectives = {
            'low': ['minor', 'slight', 'small', 'limited'],
            'medium': ['moderate', 'noticeable', 'significant', 'clear'],
            'high': ['severe', 'critical', 'major', 'extensive']
        }
    
    def generate_report(self, defect_type, confidence, sensor_data=None, wafer_id=None):
        """Generate a complete defect analysis report"""
        
        defect_key = defect_type.lower().replace(' ', '_').replace('-', '_')
        if defect_key not in self.templates:
            defect_key = 'particle'
        
        template = self.templates[defect_key]
        
        if confidence > 0.85:
            severity = 'high'
        elif confidence > 0.65:
            severity = 'medium'
        else:
            severity = 'low'
        
        severity_adj = random.choice(self.severity_adjectives[severity])
        
        report = {
            'wafer_id': wafer_id or f"WFR-{random.randint(1000, 9999)}",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'defect_type': defect_type.title(),
            'confidence': f"{confidence:.1%}",
            'severity': severity.upper(),
            'summary': self._generate_summary(defect_type, severity_adj, confidence),
            'root_cause_analysis': self._generate_rca(template, severity_adj),
            'impact_assessment': self._generate_impact(template, severity),
            'recommended_actions': self._generate_actions(template, severity),
            'sensor_analysis': self._generate_sensor_analysis(sensor_data) if sensor_data else None
        }
        
        return report
    
    def _generate_summary(self, defect_type, severity_adj, confidence):
        summaries = [
            f"A {severity_adj} {defect_type.lower()} defect has been detected with {confidence:.1%} confidence.",
            f"Visual inspection identified {severity_adj} {defect_type.lower()} pattern on the wafer surface.",
            f"The AI system detected a {severity_adj} {defect_type.lower()} anomaly requiring attention.",
        ]
        return random.choice(summaries)
    
    def _generate_rca(self, template, severity_adj):
        causes = random.sample(template['causes'], min(2, len(template['causes'])))
        rca = f"Based on the defect pattern and sensor correlation, the most likely root causes are:\n"
        for i, cause in enumerate(causes, 1):
            rca += f"  {i}. {cause.capitalize()}\n"
        return rca.strip()
    
    def _generate_impact(self, template, severity):
        impact = random.choice(template['impacts'])
        severity_impact = {
            'low': "This defect has limited impact on overall yield.",
            'medium': "This defect may affect yield in the affected region.",
            'high': "This defect poses significant risk to wafer yield and requires immediate action."
        }
        return f"{impact.capitalize()}. {severity_impact[severity]}"
    
    def _generate_actions(self, template, severity):
        num_actions = {'low': 2, 'medium': 3, 'high': 4}[severity]
        actions = random.sample(template['actions'], min(num_actions, len(template['actions'])))
        return actions
    
    def _generate_sensor_analysis(self, sensor_data):
        if not sensor_data:
            return None
        
        analysis = []
        
        if 'temperature' in sensor_data:
            temp = sensor_data['temperature']
            if temp > 500:
                analysis.append(f"⚠️ Temperature ({temp}°C) is elevated - may contribute to edge effects")
            elif temp < 400:
                analysis.append(f"⚠️ Temperature ({temp}°C) is low - check heating elements")
            else:
                analysis.append(f"✅ Temperature ({temp}°C) is within normal range")
        
        if 'pressure' in sensor_data:
            pressure = sensor_data['pressure']
            if pressure < 95:
                analysis.append(f"⚠️ Pressure ({pressure} Pa) is low - possible vacuum leak")
            elif pressure > 105:
                analysis.append(f"⚠️ Pressure ({pressure} Pa) is high - check gas flow")
            else:
                analysis.append(f"✅ Pressure ({pressure} Pa) is within normal range")
        
        if 'flow_rate' in sensor_data:
            flow = sensor_data['flow_rate']
            if abs(flow - 50) > 5:
                analysis.append(f"⚠️ Flow rate ({flow} sccm) deviation detected")
            else:
                analysis.append(f"✅ Flow rate ({flow} sccm) is nominal")
        
        return analysis
    
    def format_report_text(self, report):
        text = f"""
═══════════════════════════════════════════════════════════════
                    DEFECT ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

📋 WAFER ID: {report['wafer_id']}
🕐 TIMESTAMP: {report['timestamp']}
🔬 DEFECT TYPE: {report['defect_type']}
📊 CONFIDENCE: {report['confidence']}
⚠️ SEVERITY: {report['severity']}

───────────────────────────────────────────────────────────────
📝 SUMMARY
───────────────────────────────────────────────────────────────
{report['summary']}

───────────────────────────────────────────────────────────────
🔍 ROOT CAUSE ANALYSIS
───────────────────────────────────────────────────────────────
{report['root_cause_analysis']}

───────────────────────────────────────────────────────────────
💥 IMPACT ASSESSMENT
───────────────────────────────────────────────────────────────
{report['impact_assessment']}

───────────────────────────────────────────────────────────────
🔧 RECOMMENDED ACTIONS
───────────────────────────────────────────────────────────────
"""
        for i, action in enumerate(report['recommended_actions'], 1):
            text += f"  {i}. {action}\n"
        
        if report.get('sensor_analysis'):
            text += """
───────────────────────────────────────────────────────────────
📡 SENSOR CORRELATION
───────────────────────────────────────────────────────────────
"""
            for item in report['sensor_analysis']:
                text += f"  {item}\n"
        
        text += """
═══════════════════════════════════════════════════════════════
                Generated by Virtual Metrology AI v3.0
═══════════════════════════════════════════════════════════════
"""
        return text


if __name__ == "__main__":
    generator = DefectReportGenerator()
    
    report = generator.generate_report(
        defect_type="scratch",
        confidence=0.87,
        sensor_data={
            'temperature': 520,
            'pressure': 98,
            'flow_rate': 52
        },
        wafer_id="WFR-2024-1213-001"
    )
    
    print(generator.format_report_text(report))
