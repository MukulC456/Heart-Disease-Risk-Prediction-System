import { Component } from '@angular/core';
import { calculateRisk } from './logic/risk-calculator';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  patient = {
    age: 50,
    restingBp: 130,
    cholesterol: 220,
    fastingBloodSugar: 0,
    exerciseInducedAngina: 0,
    stDepression: 1.5,
    numMajorVessels: 0
  };

  result: string | null = null;

  submit() {
    this.result = calculateRisk(this.patient);
  }
}