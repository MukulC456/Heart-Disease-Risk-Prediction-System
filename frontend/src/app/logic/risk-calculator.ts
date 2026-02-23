export function calculateRisk(data: any): string {
    let score = 0;
  
    if (data.age > 50) score++;
    if (data.restingBp > 140) score++;
    if (data.cholesterol > 240) score++;
    if (data.fastingBloodSugar === 1) score++;
    if (data.exerciseInducedAngina === 1) score++;
    if (data.stDepression > 2.0) score++;
    if (data.numMajorVessels >= 2) score++;
  
    return score >= 3 ? 'High Risk' : 'Low Risk';
  }